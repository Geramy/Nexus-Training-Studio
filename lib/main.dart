import 'dart:io';

import 'package:flutter/material.dart';
import 'package:yaml/yaml.dart';

import 'model_scanner.dart';
import 'pipeline.dart';
import 'studio_server.dart';

/// Studio root on disk. Override with the STUDIO_ROOT env var if you move it.
final String kStudioRoot = Platform.environment['STUDIO_ROOT'] ??
    '/Users/geramyloveless/Documents/Development/Nexus-Project/Client/training_studio';

void main() => runApp(const StudioApp());

class StudioApp extends StatelessWidget {
  const StudioApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Nexus Training Studio',
        debugShowCheckedModeBanner: false,
        theme: ThemeData.dark(useMaterial3: true),
        home: const StudioHome(),
      );
}

class StudioHome extends StatefulWidget {
  const StudioHome({super.key});
  @override
  State<StudioHome> createState() => _StudioHomeState();
}

class _StudioHomeState extends State<StudioHome> {
  final List<String> _log = [];
  final ScrollController _logScroll = ScrollController();
  final _hfToken = TextEditingController();
  final _dlRepo = TextEditingController();
  final _ulSrc = TextEditingController();
  final _ulDest = TextEditingController();
  bool _ulPrivate = true;

  List<ModelInfo> _models = [];
  String _status = 'idle';

  // Config
  String _baseModel = '(unset)';
  String _python = 'python3';
  String _llama = '';
  String _quant = 'Q4_K_M';
  int _port = 8443;
  List<String> _extraDirs = const [];

  late ModelScanner _scanner;
  late Pipeline _pipeline;
  StudioServer? _server;

  @override
  void initState() {
    super.initState();
    _init();
  }

  void _addLog(String line) {
    setState(() {
      _log.add(line);
      if (_log.length > 1500) _log.removeRange(0, _log.length - 1500);
    });
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_logScroll.hasClients) {
        _logScroll.jumpTo(_logScroll.position.maxScrollExtent);
      }
    });
  }

  Future<void> _init() async {
    _loadConfig();
    _scanner = ModelScanner(extraDirs: _extraDirs);
    _pipeline = Pipeline(
      studioRoot: kStudioRoot,
      configPython: _python,
      baseModel: _baseModel,
      llamaCppDir: _llama,
      quant: _quant,
      onLog: _addLog,
    );
    _hfToken.text = _pipeline.hfToken ?? '';
    _dlRepo.text = _baseModel;
    _ulSrc.text = '$kStudioRoot/workspace/gguf';
    _server = StudioServer(
      studioRoot: kStudioRoot,
      port: _port,
      scanner: _scanner,
      statusGetter: () => _status,
      onLog: _addLog,
    );
    await _server!.start();
    await _refreshModels();
    _addLog('Studio ready. Base model: $_baseModel  ·  Python env: '
        '${_pipeline.envReady ? "ready" : "NOT set up — click Setup env"}');
  }

  void _loadConfig() {
    try {
      final y = loadYaml(File('$kStudioRoot/config.yaml').readAsStringSync())
          as YamlMap;
      _baseModel = (y['base_model'] ?? _baseModel).toString();
      _python = (y['python'] ?? _python).toString();
      _llama = (y['llama_cpp_dir'] ?? _llama).toString();
      _quant = (y['quant'] ?? _quant).toString();
      _port = int.tryParse('${y['port'] ?? _port}') ?? _port;
      final extra = y['extra_model_dirs'];
      if (extra is YamlList) _extraDirs = [for (final e in extra) e.toString()];
    } catch (e) {
      _addLog('config.yaml not loaded ($e) — using defaults.');
    }
  }

  Future<void> _refreshModels() async {
    final m = await _scanner.scan();
    setState(() => _models = m);
    _addLog('Scanned ${m.length} local model(s).');
  }

  Future<void> _step(String status, Future<dynamic> Function() fn) async {
    setState(() => _status = status);
    await fn();
    setState(() => _status = 'idle');
  }

  Future<void> _deleteModel(ModelInfo m) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Delete model?'),
        content: Text('${m.name}\n${m.path}\n(${m.sizeGb.toStringAsFixed(1)} GB) '
            '— this permanently removes it from disk.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Delete')),
        ],
      ),
    );
    if (ok != true) return;
    try {
      await deleteModel(m);
      _addLog('Deleted ${m.name}');
    } catch (e) {
      _addLog('Delete failed: $e');
    }
    await _refreshModels();
  }

  int _rawCount() {
    final f = File('$kStudioRoot/workspace/data/raw.jsonl');
    if (!f.existsSync()) return 0;
    return f.readAsLinesSync().where((l) => l.trim().isNotEmpty).length;
  }

  @override
  void dispose() {
    _server?.stop();
    _logScroll.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final busy = _pipeline.running;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Nexus Training Studio'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(child: Text('status: $_status')),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _headerCard(),
            const SizedBox(height: 8),
            _hfCard(busy),
            const SizedBox(height: 8),
            Wrap(spacing: 8, runSpacing: 8, children: [
              _btn('Setup env', Icons.terminal,
                  busy ? null : () => _step('setup-env', _pipeline.setupEnv)),
              _btn('Check support', Icons.fact_check_outlined,
                  busy ? null : () => _step('checking', _pipeline.checkSupport)),
              _btn('Prepare data', Icons.dataset_outlined,
                  busy ? null : () => _step('preparing', _pipeline.prepareData)),
              _btn('Train (LoRA)', Icons.model_training_outlined,
                  busy ? null : () => _step('training', _pipeline.train)),
              _btn('Eval', Icons.checklist_outlined,
                  busy ? null : () => _step('evaluating', _pipeline.evaluate)),
              _btn('Export GGUF', Icons.save_alt_outlined,
                  busy ? null : () => _step('exporting', _pipeline.exportGguf)),
              _btn('Cancel', Icons.stop_circle_outlined,
                  busy ? () => _pipeline.cancel() : null),
              _btn('Rescan', Icons.refresh, _refreshModels),
            ]),
            const SizedBox(height: 10),
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SizedBox(width: 340, child: _modelsPanel(busy)),
                  const SizedBox(width: 10),
                  Expanded(child: _logConsole()),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _headerCard() => Card(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Base model: $_baseModel',
                  style: const TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 4),
              Text('API: ${_server?.baseUrl ?? "(starting…)"}   ·   '
                  'traces received: ${_rawCount()}   ·   '
                  'env: ${_pipeline.envReady ? "ready" : "not set up"}'),
              const SizedBox(height: 2),
              Text('GGUF quant: $_quant   ·   llama.cpp: '
                  '${_llama.isEmpty ? "(set in config.yaml)" : _llama}',
                  style: const TextStyle(fontSize: 12, color: Colors.white70)),
            ],
          ),
        ),
      );

  Widget _hfCard(bool busy) => Card(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Hugging Face',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              Row(children: [
                Expanded(
                  child: TextField(
                    controller: _hfToken,
                    obscureText: true,
                    decoration: const InputDecoration(
                      isDense: true,
                      labelText: 'HF token (for private repos + uploads)',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton(
                    onPressed: () => _pipeline.saveToken(_hfToken.text),
                    child: const Text('Save token')),
              ]),
              const SizedBox(height: 8),
              Row(children: [
                Expanded(
                  child: TextField(
                    controller: _dlRepo,
                    decoration: const InputDecoration(
                      isDense: true,
                      labelText: 'Download repo (org/name)',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                FilledButton.tonalIcon(
                  onPressed: busy
                      ? null
                      : () => _step('downloading',
                          () => _pipeline.downloadModel(_dlRepo.text.trim())),
                  icon: const Icon(Icons.download, size: 18),
                  label: const Text('Download'),
                ),
              ]),
              const SizedBox(height: 8),
              Row(children: [
                Expanded(
                  flex: 3,
                  child: TextField(
                    controller: _ulSrc,
                    decoration: const InputDecoration(
                      isDense: true,
                      labelText: 'Upload source (folder or file)',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  flex: 3,
                  child: TextField(
                    controller: _ulDest,
                    decoration: const InputDecoration(
                      isDense: true,
                      labelText: 'Dest repo (user/name or org/name)',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Row(children: [
                  Switch(
                      value: _ulPrivate,
                      onChanged: (v) => setState(() => _ulPrivate = v)),
                  const Text('private'),
                ]),
                const SizedBox(width: 8),
                FilledButton.tonalIcon(
                  onPressed: busy
                      ? null
                      : () => _step(
                          'uploading',
                          () => _pipeline.uploadModel(_ulSrc.text.trim(),
                              _ulDest.text.trim(), _ulPrivate)),
                  icon: const Icon(Icons.upload, size: 18),
                  label: const Text('Upload'),
                ),
              ]),
            ],
          ),
        ),
      );

  Widget _btn(String label, IconData icon, VoidCallback? onTap) =>
      FilledButton.tonalIcon(
          onPressed: onTap, icon: Icon(icon, size: 18), label: Text(label));

  Widget _modelsPanel(bool busy) => Card(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Padding(
              padding: EdgeInsets.all(10),
              child: Text('Local models',
                  style: TextStyle(fontWeight: FontWeight.bold)),
            ),
            const Divider(height: 1),
            Expanded(
              child: _models.isEmpty
                  ? const Center(
                      child: Text('No models found.\nDownload one above,\n'
                          'then Rescan.', textAlign: TextAlign.center))
                  : ListView.builder(
                      itemCount: _models.length,
                      itemBuilder: (_, i) {
                        final m = _models[i];
                        return ListTile(
                          dense: true,
                          title: Text(m.name,
                              maxLines: 1, overflow: TextOverflow.ellipsis),
                          subtitle: Text('${m.source} · ${m.format} · '
                              '${m.sizeGb.toStringAsFixed(1)} GB'),
                          trailing: IconButton(
                            icon: const Icon(Icons.delete_outline, size: 18),
                            tooltip: 'Delete from disk',
                            onPressed: busy ? null : () => _deleteModel(m),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      );

  Widget _logConsole() => Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
            color: Colors.black, borderRadius: BorderRadius.circular(8)),
        child: ListView.builder(
          controller: _logScroll,
          itemCount: _log.length,
          itemBuilder: (_, i) => Text(
            _log[i],
            style: const TextStyle(
                fontFamily: 'monospace', fontSize: 12, color: Color(0xFFB9F6CA)),
          ),
        ),
      );
}
