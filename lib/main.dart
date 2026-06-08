import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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

/// One workflow stage = one page in the left nav.
class _NavStage {
  final IconData icon;
  final String label;
  const _NavStage(this.icon, this.label);
}

const _stages = <_NavStage>[
  _NavStage(Icons.dashboard_outlined, 'Overview'),
  _NavStage(Icons.terminal, '1 · Environment'),
  _NavStage(Icons.cloud_download_outlined, '2 · Base Model'),
  _NavStage(Icons.dataset_outlined, '3 · Data'),
  _NavStage(Icons.compress, '4 · Quantize Base'),
  _NavStage(Icons.model_training_outlined, '5 · Train'),
  _NavStage(Icons.checklist_outlined, '6 · Evaluate'),
  _NavStage(Icons.science_outlined, '7 · Test Cases'),
  _NavStage(Icons.save_alt_outlined, '8 · Export GGUF'),
  _NavStage(Icons.cloud_upload_outlined, '9 · Upload'),
  _NavStage(Icons.storage_outlined, 'Models'),
];

class StudioHome extends StatefulWidget {
  const StudioHome({super.key});
  @override
  State<StudioHome> createState() => _StudioHomeState();
}

class _StudioHomeState extends State<StudioHome> {
  final List<String> _log = [];
  final ScrollController _logScroll = ScrollController();
  final _hfToken = TextEditingController();
  final _ulSrc = TextEditingController();
  final _ulDest = TextEditingController();
  final _newRepoName = TextEditingController();
  final _testModel = TextEditingController(text: 'workspace/fused');
  final _excelPath = TextEditingController();
  final _hfDataset = TextEditingController();
  final _hfSplit = TextEditingController(text: 'train');
  List<Map<String, dynamic>> _dataRows = [];
  int _dataTotal = 0;
  String _dataFilter = 'all';

  // Known academic / community datasets for requirements + user stories.
  static const _hfSuggestions = <String>[
    'nguyen-brat/user-story',
    'Salesforce/xlam-function-calling-60k',
    'glaiveai/glaive-function-calling-v2',
  ];
  bool _ulPrivate = true;
  bool _creatingRepo = false;
  String? _newRepoNs;

  // Download field (HF search autocomplete) controller, captured from the widget.
  TextEditingController? _dlCtl;
  TextEditingController? _dsCtl;
  Timer? _searchDebounce;
  Timer? _dsDebounce;

  List<String> _repos = [];
  List<String> _namespaces = [];

  List<ModelInfo> _models = [];
  String _status = 'idle';
  int _page = 0;
  bool _logExpanded = true;

  // Config
  String _baseModel = '(unset)';
  String _python = 'python3';
  String _llama = '';
  int _trainBits = 8;
  String _exportQuants = 'Q8_0,Q6_K,Q4_K_M';
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
      trainBits: _trainBits,
      exportQuants: _exportQuants,
      onLog: _addLog,
    );
    _hfToken.text = _pipeline.hfToken ?? '';
    _ulSrc.text = '$kStudioRoot/workspace/gguf';
    _server = StudioServer(
      studioRoot: kStudioRoot,
      port: _port,
      scanner: _scanner,
      pipeline: _pipeline,
      logsTail: () => List<String>.from(_log),
      onStage: (s) {
        if (mounted) setState(() => _status = s);
      },
      onLog: _addLog,
    );
    await _server!.start();
    await _refreshModels();
    _loadData();
    _addLog('Studio ready. Base model: $_baseModel  ·  Python env: '
        '${_pipeline.envReady ? "ready" : "NOT set up — go to step 1"}');
  }

  void _loadConfig() {
    try {
      final y = loadYaml(File('$kStudioRoot/config.yaml').readAsStringSync())
          as YamlMap;
      _baseModel = (y['base_model'] ?? _baseModel).toString();
      _python = (y['python'] ?? _python).toString();
      _llama = (y['llama_cpp_dir'] ?? _llama).toString();
      _trainBits = int.tryParse('${y['train_bits'] ?? _trainBits}') ?? _trainBits;
      _exportQuants = (y['export_quants'] ?? _exportQuants).toString();
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

  void _loadData() {
    setState(() {
      _dataRows = _pipeline.datasetSummary(limit: 500);
      _dataTotal = _pipeline.datasetCount();
    });
  }

  Future<void> _stepThenReloadData(
      String status, Future<dynamic> Function() fn) async {
    await _step(status, fn);
    _loadData();
  }

  void _deleteRow(String id) {
    if (_pipeline.deleteConversation(id)) _addLog('- deleted $id');
    _loadData();
  }

  Future<void> _addRowDialog() async {
    final ctl = TextEditingController(
        text: '[\n'
            '  {"role": "system", "content": "You are…"},\n'
            '  {"role": "user", "content": "…"},\n'
            '  {"role": "assistant", "content": "…"}\n'
            ']');
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Add conversation (JSON messages[])'),
        content: SizedBox(
          width: 560,
          child: TextField(
            controller: ctl,
            maxLines: 14,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            decoration: const InputDecoration(border: OutlineInputBorder()),
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Add')),
        ],
      ),
    );
    if (ok != true) return;
    try {
      final parsed = jsonDecode(ctl.text);
      if (parsed is! List) throw 'top level must be a JSON array of messages';
      final added = _pipeline.addConversation(parsed, source: 'manual');
      _addLog(added ? '+ manual row added' : '· rejected (dup/invalid)');
      _loadData();
    } catch (e) {
      _addLog('Add failed: $e');
    }
  }

  Future<void> _loadRepos() async {
    _addLog('Loading your HF repos…');
    final r = await _pipeline.listRepos();
    if (r['error'] != null) {
      _addLog('HF repos: ${r['error']}');
      return;
    }
    setState(() {
      _namespaces = [for (final n in (r['namespaces'] ?? [])) n.toString()];
      _repos = [for (final n in (r['repos'] ?? [])) n.toString()];
      _newRepoNs ??= _namespaces.isNotEmpty ? _namespaces.first : null;
    });
    _addLog('Loaded ${_repos.length} repo(s) across '
        '${_namespaces.length} namespace(s): ${_namespaces.join(", ")}');
  }

  /// Debounced HF search for the model download autocomplete.
  Future<Iterable<String>> _debouncedSearch(String q) {
    _searchDebounce?.cancel();
    if (q.trim().length < 3) {
      return Future.value(const Iterable<String>.empty());
    }
    final c = Completer<Iterable<String>>();
    _searchDebounce = Timer(const Duration(milliseconds: 450), () async {
      if (!c.isCompleted) c.complete(await _pipeline.searchModels(q.trim()));
    });
    return c.future;
  }

  /// Debounced HF search for the dataset-import autocomplete.
  Future<Iterable<String>> _debouncedDatasetSearch(String q) {
    _dsDebounce?.cancel();
    if (q.trim().length < 3) {
      return Future.value(const Iterable<String>.empty());
    }
    final c = Completer<Iterable<String>>();
    _dsDebounce = Timer(const Duration(milliseconds: 450), () async {
      if (!c.isCompleted) c.complete(await _pipeline.searchDatasets(q.trim()));
    });
    return c.future;
  }

  // ── Native path pickers (file / folder / save) ──
  Future<void> _pickFileInto(TextEditingController ctl,
      {List<XTypeGroup> groups = const []}) async {
    final f = await openFile(acceptedTypeGroups: groups);
    if (f != null) setState(() => ctl.text = f.path);
  }

  Future<void> _pickDirInto(TextEditingController ctl) async {
    final d = await getDirectoryPath();
    if (d != null) setState(() => ctl.text = d);
  }

  Widget _browseBtn(VoidCallback onTap, {String tip = 'Browse'}) => IconButton(
        icon: const Icon(Icons.folder_open, size: 20),
        tooltip: tip,
        onPressed: onTap,
      );

  Future<void> _step(String status, Future<dynamic> Function() fn) async {
    setState(() => _status = status);
    await fn();
    setState(() => _status = 'idle');
  }

  void _syncNewRepo() {
    final ns = _newRepoNs ?? '';
    final name = _newRepoName.text.trim();
    _ulDest.text = (ns.isEmpty || name.isEmpty) ? name : '$ns/$name';
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
    _searchDebounce?.cancel();
    _dsDebounce?.cancel();
    _server?.stop();
    _logScroll.dispose();
    super.dispose();
  }

  // ════════════════════════════ BUILD ════════════════════════════
  @override
  Widget build(BuildContext context) {
    final busy = _pipeline.running;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Nexus Training Studio'),
        actions: [
          if (busy)
            const Padding(
              padding: EdgeInsets.only(right: 8),
              child: Center(
                child: SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2)),
              ),
            ),
          Padding(
            padding: const EdgeInsets.only(right: 4),
            child: Center(
              child: Chip(
                  label: Text('status: $_status'),
                  visualDensity: VisualDensity.compact),
            ),
          ),
          IconButton(
            tooltip: 'Cancel running step',
            icon: const Icon(Icons.stop_circle_outlined),
            onPressed: busy ? () => _pipeline.cancel() : null,
          ),
          IconButton(
            tooltip: 'Rescan models',
            icon: const Icon(Icons.refresh),
            onPressed: _refreshModels,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _sidebar(),
          const VerticalDivider(width: 1),
          // ── Page + persistent log dock ──
          Expanded(
            child: Column(
              children: [
                Expanded(child: _pageBody(busy)),
                _logDock(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ── Left navigation ──
  Widget _sidebar() => SizedBox(
        width: 210,
        child: Column(
          children: [
            Expanded(
              child: ListView.builder(
                padding: const EdgeInsets.symmetric(vertical: 8),
                itemCount: _stages.length,
                itemBuilder: (_, i) {
                  final s = _stages[i];
                  final sel = i == _page;
                  return Container(
                    margin:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: sel
                          ? Theme.of(context).colorScheme.primaryContainer
                          : null,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: ListTile(
                      dense: true,
                      leading: Icon(s.icon,
                          size: 20,
                          color: sel
                              ? Theme.of(context).colorScheme.onPrimaryContainer
                              : null),
                      title: Text(s.label,
                          style: TextStyle(
                              fontSize: 13,
                              fontWeight:
                                  sel ? FontWeight.bold : FontWeight.normal)),
                      onTap: () => setState(() => _page = i),
                    ),
                  );
                },
              ),
            ),
            const Divider(height: 1),
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(_server?.baseUrl ?? '(starting…)',
                  style: const TextStyle(fontSize: 10, color: Colors.white54)),
            ),
          ],
        ),
      );

  // ── Page router ──
  Widget _pageBody(bool busy) {
    switch (_page) {
      case 0:
        return _pageOverview();
      case 1:
        return _pageEnv(busy);
      case 2:
        return _pageBaseModel(busy);
      case 3:
        return _pageData(busy);
      case 4:
        return _pageQuantize(busy);
      case 5:
        return _pageTrain(busy);
      case 6:
        return _pageEval(busy);
      case 7:
        return _pageTest(busy);
      case 8:
        return _pageExport(busy);
      case 9:
        return _pageUpload(busy);
      case 10:
        return _pageModels(busy);
      default:
        return _pageOverview();
    }
  }

  /// Standard page chrome: title, subtitle, body, and Back/Next footer.
  Widget _page_(String title, String subtitle, List<Widget> body,
          {bool footer = true}) =>
      Padding(
        padding: const EdgeInsets.fromLTRB(20, 18, 20, 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(title,
                style: const TextStyle(
                    fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 2),
            Text(subtitle,
                style: const TextStyle(color: Colors.white70, fontSize: 13)),
            const SizedBox(height: 16),
            Expanded(child: ListView(children: body)),
            if (footer) _navFooter(),
          ],
        ),
      );

  Widget _navFooter() => Padding(
        padding: const EdgeInsets.only(top: 8),
        child: Row(
          children: [
            TextButton.icon(
              onPressed:
                  _page > 0 ? () => setState(() => _page -= 1) : null,
              icon: const Icon(Icons.arrow_back, size: 18),
              label: const Text('Back'),
            ),
            const Spacer(),
            FilledButton.icon(
              onPressed: _page < _stages.length - 1
                  ? () => setState(() => _page += 1)
                  : null,
              icon: const Icon(Icons.arrow_forward, size: 18),
              label: const Text('Next'),
            ),
          ],
        ),
      );

  // ════════════════════════════ PAGES ════════════════════════════

  Widget _pageOverview() => _page_(
        'Overview',
        'Full pipeline: env → base model → data → quantize → train → '
            'eval → test → export → upload.',
        [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Wrap(spacing: 24, runSpacing: 10, children: [
                _kv('Base model', _baseModel),
                _kv('API', _server?.baseUrl ?? '(starting…)'),
                _kv('Python env', _pipeline.envReady ? 'ready ✓' : 'not set up'),
                _kv('8-bit base', _pipeline.base8bitReady ? 'built ✓' : 'not built'),
                _kv('Traces received', '${_rawCount()}'),
                _kv('Train precision', '${_trainBits}-bit'),
                _kv('Export quants', _exportQuants),
                _kv('Local models', '${_models.length}'),
                _kv('llama.cpp', _llama.isEmpty ? '(set in config)' : _llama),
              ]),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('How it works',
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  const Text(
                      'Train QLoRA on an 8-bit MLX base, fuse the adapter, then '
                      'export imatrix-quantized GGUFs (8/6/4-bit). Agents also '
                      'POST live traces to this app over the API — they grow your '
                      'training set automatically.',
                      style: TextStyle(fontSize: 13, height: 1.4)),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: () => setState(() => _page = 1),
                    icon: const Icon(Icons.play_arrow, size: 18),
                    label: const Text('Start at Step 1'),
                  ),
                ],
              ),
            ),
          ),
        ],
        footer: false,
      );

  Widget _pageEnv(bool busy) => _page_(
        'Step 1 · Environment',
        'Create the Python venv and install mlx-lm + tools.',
        [
          Row(children: [
            _btn('Setup env', Icons.terminal,
                busy ? null : () => _step('setup-env', _pipeline.setupEnv)),
            const SizedBox(width: 12),
            _statusDot(_pipeline.envReady, 'ready', 'not set up yet'),
          ]),
        ],
      );

  Widget _pageBaseModel(bool busy) => _page_(
        'Step 2 · Base Model',
        'Save your HF token, then search Hugging Face and download a model.',
        [
          Row(children: [
            Expanded(
              child: TextField(
                controller: _hfToken,
                obscureText: true,
                decoration: const InputDecoration(
                  isDense: true,
                  labelText: 'HF token (private repos + uploads)',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            const SizedBox(width: 8),
            FilledButton(
                onPressed: () => _pipeline.saveToken(_hfToken.text),
                child: const Text('Save')),
          ]),
          const SizedBox(height: 14),
          Row(children: [
            Expanded(
              child: Autocomplete<String>(
                optionsBuilder: (tev) => _debouncedSearch(tev.text),
                fieldViewBuilder: (ctx, controller, focus, onSubmit) {
                  if (_dlCtl != controller) {
                    _dlCtl = controller;
                    if (controller.text.isEmpty) controller.text = _baseModel;
                  }
                  return TextField(
                    controller: controller,
                    focusNode: focus,
                    onSubmitted: (_) => onSubmit(),
                    decoration: const InputDecoration(
                      isDense: true,
                      labelText: 'Search / download repo (type 3+ chars)',
                      prefixIcon: Icon(Icons.search, size: 18),
                      border: OutlineInputBorder(),
                    ),
                  );
                },
                optionsViewBuilder: (ctx, onSelected, options) => Align(
                  alignment: Alignment.topLeft,
                  child: Material(
                    elevation: 4,
                    child: ConstrainedBox(
                      constraints:
                          const BoxConstraints(maxHeight: 300, maxWidth: 560),
                      child: ListView(
                        padding: EdgeInsets.zero,
                        shrinkWrap: true,
                        children: [
                          for (final o in options)
                            ListTile(
                              dense: true,
                              title:
                                  Text(o, style: const TextStyle(fontSize: 13)),
                              onTap: () => onSelected(o),
                            ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.tonalIcon(
              onPressed: busy
                  ? null
                  : () {
                      final repo = (_dlCtl?.text ?? '').trim();
                      if (repo.isEmpty) return;
                      _step('downloading', () => _pipeline.downloadModel(repo));
                    },
              icon: const Icon(Icons.download, size: 18),
              label: const Text('Download'),
            ),
          ]),
          const SizedBox(height: 8),
          const Text('Search is debounced (450ms). Picks any HF model repo.',
              style: TextStyle(fontSize: 11, color: Colors.white54)),
        ],
      );

  Widget _pageData(bool busy) {
    final filtered = _dataFilter == 'all'
        ? _dataRows
        : _dataRows.where((r) => r['kind'] == _dataFilter).toList();
    final counts = <String, int>{};
    for (final r in _dataRows) {
      counts[r['kind'] as String] = (counts[r['kind'] as String] ?? 0) + 1;
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text('Step 3 · Data',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const Text('Build the corpus: generate, import Excel/HF, edit rows, '
              'then Prepare into train/valid.',
              style: TextStyle(color: Colors.white70, fontSize: 13)),
          const SizedBox(height: 12),

          // ── Build / import toolbar ──
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Wrap(spacing: 8, runSpacing: 8, crossAxisAlignment: WrapCrossAlignment.center, children: [
                    _btn('Generate corpus', Icons.auto_awesome,
                        busy ? null : () => _stepThenReloadData(
                            'generating', () => _pipeline.generateData())),
                    _btn('Add row', Icons.add, busy ? null : _addRowDialog),
                    _btn('Excel template', Icons.description_outlined,
                        busy ? null : () async {
                          final loc = await getSaveLocation(
                              suggestedName: 'nexus-training-template.xlsx');
                          if (loc != null) {
                            _step('template',
                                () => _pipeline.exportTemplate(loc.path));
                          }
                        }),
                    _btn('Prepare data', Icons.dataset_outlined,
                        busy ? null : () => _step('preparing',
                            _pipeline.prepareData)),
                    _btn('Refresh', Icons.refresh, _loadData),
                  ]),
                  const Divider(height: 20),
                  // Excel import row (with native file picker)
                  Row(children: [
                    Expanded(
                      child: TextField(
                        controller: _excelPath,
                        decoration: const InputDecoration(
                          isDense: true,
                          labelText: 'Excel/CSV path to import',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    _browseBtn(() => _pickFileInto(_excelPath, groups: const [
                          XTypeGroup(label: 'spreadsheets',
                              extensions: ['xlsx', 'xls', 'csv']),
                        ]), tip: 'Choose Excel/CSV'),
                    const SizedBox(width: 4),
                    FilledButton.tonalIcon(
                      onPressed: busy
                          ? null
                          : () => _stepThenReloadData('importing',
                              () => _pipeline.importExcel(_excelPath.text.trim())),
                      icon: const Icon(Icons.table_view, size: 18),
                      label: const Text('Import Excel'),
                    ),
                  ]),
                  const SizedBox(height: 8),
                  // HF dataset import row
                  Row(children: [
                    Expanded(
                      flex: 3,
                      child: Autocomplete<String>(
                        optionsBuilder: (tev) async {
                          final typed = await _debouncedDatasetSearch(tev.text);
                          // surface known good sets when the box is short
                          if (tev.text.trim().length < 3) return _hfSuggestions;
                          return typed;
                        },
                        fieldViewBuilder: (ctx, c, f, onSub) {
                          if (_dsCtl != c) {
                            _dsCtl = c;
                            if (c.text.isEmpty) c.text = _hfDataset.text;
                          }
                          return TextField(
                            controller: c,
                            focusNode: f,
                            onChanged: (v) => _hfDataset.text = v,
                            decoration: const InputDecoration(
                              isDense: true,
                              labelText: 'Search HF datasets (type 3+ chars)',
                              prefixIcon: Icon(Icons.search, size: 18),
                              border: OutlineInputBorder(),
                            ),
                          );
                        },
                        onSelected: (s) => setState(() => _hfDataset.text = s),
                        optionsViewBuilder: (ctx, onSelected, options) => Align(
                          alignment: Alignment.topLeft,
                          child: Material(
                            elevation: 4,
                            child: ConstrainedBox(
                              constraints: const BoxConstraints(
                                  maxHeight: 280, maxWidth: 460),
                              child: ListView(
                                padding: EdgeInsets.zero,
                                shrinkWrap: true,
                                children: [
                                  for (final o in options)
                                    ListTile(
                                      dense: true,
                                      title: Text(o,
                                          style: const TextStyle(fontSize: 13)),
                                      onTap: () => onSelected(o),
                                    ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    SizedBox(
                      width: 110,
                      child: TextField(
                        controller: _hfSplit,
                        decoration: const InputDecoration(
                          isDense: true,
                          labelText: 'split',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    FilledButton.tonalIcon(
                      onPressed: busy
                          ? null
                          : () => _stepThenReloadData('importing', () =>
                              _pipeline.importHf(_hfDataset.text.trim(),
                                  split: _hfSplit.text.trim().isEmpty
                                      ? 'train'
                                      : _hfSplit.text.trim())),
                      icon: const Icon(Icons.cloud_download, size: 18),
                      label: const Text('Import HF'),
                    ),
                  ]),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // ── Filter chips + count ──
          Row(children: [
            Text(
                _dataTotal > _dataRows.length
                    ? '$_dataTotal total · showing ${_dataRows.length}'
                    : '${_dataRows.length} conversation(s)',
                style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(width: 12),
            for (final k in ['all', 'setup', 'discovery', 'tasks', 'other'])
              Padding(
                padding: const EdgeInsets.only(right: 6),
                child: FilterChip(
                  label: Text(k == 'all'
                      ? 'all (${_dataRows.length})'
                      : '$k (${counts[k] ?? 0})'),
                  selected: _dataFilter == k,
                  onSelected: (_) => setState(() => _dataFilter = k),
                ),
              ),
          ]),
          const SizedBox(height: 8),

          // ── The table ──
          Expanded(child: _dataTable(filtered, busy)),
          _navFooter(),
        ],
      ),
    );
  }

  Widget _dataTable(List<Map<String, dynamic>> rows, bool busy) {
    if (rows.isEmpty) {
      return const Card(
        child: Center(
          child: Text('No rows yet.\nGenerate the corpus or import Excel/HF.',
              textAlign: TextAlign.center),
        ),
      );
    }
    return Card(
      child: Column(
        children: [
          // header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: const BoxDecoration(
                border: Border(bottom: BorderSide(color: Colors.white24))),
            child: Row(children: const [
              SizedBox(width: 78, child: Text('Kind',
                  style: TextStyle(fontWeight: FontWeight.bold))),
              SizedBox(width: 120, child: Text('Source',
                  style: TextStyle(fontWeight: FontWeight.bold))),
              SizedBox(width: 48, child: Text('Turns',
                  style: TextStyle(fontWeight: FontWeight.bold))),
              SizedBox(width: 44, child: Text('Tools',
                  style: TextStyle(fontWeight: FontWeight.bold))),
              Expanded(child: Text('First user message / tool calls',
                  style: TextStyle(fontWeight: FontWeight.bold))),
              SizedBox(width: 40),
            ]),
          ),
          Expanded(
            child: ListView.separated(
              itemCount: rows.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (_, i) {
                final r = rows[i];
                final calls = (r['calls'] as List).cast<String>();
                return Padding(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(width: 78, child: _kindChip(r['kind'] as String)),
                      SizedBox(
                          width: 120,
                          child: Text('${r['source']}',
                              style: const TextStyle(fontSize: 12),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis)),
                      SizedBox(
                          width: 48,
                          child: Text('${r['turns']}',
                              style: const TextStyle(fontSize: 12))),
                      SizedBox(
                          width: 44,
                          child: Text('${r['tools']}',
                              style: const TextStyle(fontSize: 12))),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('${r['preview']}',
                                style: const TextStyle(fontSize: 12)),
                            if (calls.isNotEmpty)
                              Padding(
                                padding: const EdgeInsets.only(top: 2),
                                child: Text(calls.join(' · '),
                                    style: const TextStyle(
                                        fontSize: 11,
                                        color: Colors.lightBlueAccent)),
                              ),
                          ],
                        ),
                      ),
                      SizedBox(
                        width: 40,
                        child: IconButton(
                          icon: const Icon(Icons.delete_outline, size: 18),
                          tooltip: 'Delete row',
                          onPressed: busy
                              ? null
                              : () => _deleteRow('${r['id']}'),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _kindChip(String kind) {
    final color = {
      'setup': Colors.teal,
      'discovery': Colors.indigo,
      'tasks': Colors.deepOrange,
    }[kind] ?? Colors.grey;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
          color: color.withValues(alpha: 0.3),
          borderRadius: BorderRadius.circular(6)),
      child: Text(kind, style: const TextStyle(fontSize: 11)),
    );
  }

  Widget _pageQuantize(bool busy) => _page_(
        'Step 4 · Quantize Base → ${_trainBits}-bit',
        'Build the ${_trainBits}-bit MLX base that QLoRA trains on.',
        [
          Row(children: [
            _btn('Quantize 8-bit', Icons.compress,
                busy ? null : () => _step('quantize-base', _pipeline.quantizeBase)),
            const SizedBox(width: 12),
            _statusDot(_pipeline.base8bitReady, 'base ready', 'not built yet'),
          ]),
        ],
      );

  Widget _pageTrain(bool busy) => _page_(
        'Step 5 · Train (LoRA)',
        'QLoRA fine-tune on the ${_trainBits}-bit base with your data.',
        [
          Row(children: [
            _btn('Train (LoRA)', Icons.model_training_outlined,
                busy ? null : () => _step('training', _pipeline.train)),
          ]),
        ],
      );

  Widget _pageEval(bool busy) => _page_(
        'Step 6 · Evaluate',
        'Sanity-check the fused model loss before exporting.',
        [
          Row(children: [
            _btn('Eval', Icons.checklist_outlined,
                busy ? null : () => _step('evaluating', _pipeline.evaluate)),
          ]),
        ],
      );

  Widget _pageTest(bool busy) => _page_(
        'Step 7 · Test Cases',
        'Run the fine-tuned model on your use cases and report pass/fail.',
        [
          Wrap(
              spacing: 8,
              runSpacing: 4,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                const Text('Model:', style: TextStyle(color: Colors.white70)),
                ActionChip(
                  label: const Text('Fused fine-tune'),
                  onPressed: () =>
                      setState(() => _testModel.text = 'workspace/fused'),
                ),
                for (final m in _models.where((m) => m.format == 'gguf'))
                  ActionChip(
                    label: Text(m.name, overflow: TextOverflow.ellipsis),
                    onPressed: () => setState(() => _testModel.text = m.path),
                  ),
              ]),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(
              child: TextField(
                controller: _testModel,
                decoration: const InputDecoration(
                  isDense: true,
                  labelText: 'Model to test (dir or .gguf)',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            _browseBtn(() => _pickDirInto(_testModel), tip: 'Choose model folder'),
            _browseBtn(
                () => _pickFileInto(_testModel, groups: const [
                      XTypeGroup(label: 'gguf', extensions: ['gguf'])
                    ]),
                tip: 'Choose .gguf file'),
            const SizedBox(width: 4),
            _btn('Run test cases', Icons.science_outlined,
                busy
                    ? null
                    : () => _step('testing',
                        () => _pipeline.runTests(model: _testModel.text.trim()))),
            const SizedBox(width: 4),
            _btn('Eval tool-calls', Icons.rule,
                busy
                    ? null
                    : () => _step('eval-tools', () => _pipeline.evalToolCalls(
                        model: _testModel.text.trim()))),
          ]),
          const SizedBox(height: 8),
          const Text('"Eval tool-calls" scores name + JSON-args exact-match on '
              'held-out valid.jsonl (BFCL-style). Edit cases in '
              'workspace/tests/cases.jsonl — '
              '{"name","prompt","expect":[...substrings]}. '
              'A starter file is created on first run.',
              style: TextStyle(fontSize: 12, color: Colors.white54)),
        ],
      );

  Widget _pageExport(bool busy) => _page_(
        'Step 8 · Export GGUF',
        'Fuse adapter → GGUF, imatrix-quantize to: $_exportQuants',
        [
          Row(children: [
            _btn('Export GGUF', Icons.save_alt_outlined,
                busy ? null : () => _step('exporting', _pipeline.exportGguf)),
          ]),
          const SizedBox(height: 8),
          const Text('Uses an importance matrix from your training data for the '
              'best-quality K-quants (the stock equivalent of "_XL" quants).',
              style: TextStyle(fontSize: 12, color: Colors.white54)),
        ],
      );

  Widget _pageUpload(bool busy) => _page_(
        'Step 9 · Upload to Hugging Face',
        'Pick or create a repo, then push the quantized + original model.',
        [
          Row(children: [
            FilledButton.tonalIcon(
              onPressed: busy ? null : _loadRepos,
              icon: const Icon(Icons.cloud_sync, size: 18),
              label: const Text('Load my repos'),
            ),
            const SizedBox(width: 8),
            Expanded(child: _destRepoField()),
          ]),
          if (_creatingRepo) ...[
            const SizedBox(height: 10),
            _newRepoRow(),
          ],
          const SizedBox(height: 14),
          Wrap(
              spacing: 8,
              runSpacing: 4,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                const Text('Source:', style: TextStyle(color: Colors.white70)),
                ActionChip(
                  avatar: const Icon(Icons.compress, size: 16),
                  label: const Text('Quantized GGUFs'),
                  onPressed: () => setState(
                      () => _ulSrc.text = '$kStudioRoot/workspace/gguf'),
                ),
                ActionChip(
                  avatar: const Icon(Icons.inventory_2_outlined, size: 16),
                  label: const Text('Original (fused f16)'),
                  onPressed: () => setState(
                      () => _ulSrc.text = '$kStudioRoot/workspace/fused'),
                ),
                ActionChip(
                  avatar: const Icon(Icons.layers_outlined, size: 16),
                  label: const Text('Adapters'),
                  onPressed: () => setState(
                      () => _ulSrc.text = '$kStudioRoot/workspace/adapters'),
                ),
              ]),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(
              child: TextField(
                controller: _ulSrc,
                decoration: const InputDecoration(
                  isDense: true,
                  labelText: 'Upload source (folder or file)',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
            _browseBtn(() => _pickDirInto(_ulSrc), tip: 'Choose folder'),
            _browseBtn(() => _pickFileInto(_ulSrc), tip: 'Choose file'),
          ]),
          const SizedBox(height: 12),
          Row(children: [
            Row(children: [
              Switch(
                  value: _ulPrivate,
                  onChanged: (v) => setState(() => _ulPrivate = v)),
              const Text('private'),
            ]),
            const Spacer(),
            FilledButton.icon(
              onPressed: busy
                  ? null
                  : () {
                      final dest = _ulDest.text.trim();
                      if (dest.isEmpty) {
                        _addLog('Pick or create a destination repo first.');
                        return;
                      }
                      _step(
                          'uploading',
                          () => _pipeline.uploadModel(
                              _ulSrc.text.trim(), dest, _ulPrivate));
                    },
              icon: const Icon(Icons.upload, size: 18),
              label: Text(
                  _ulDest.text.isEmpty ? 'Upload' : 'Upload → ${_ulDest.text}'),
            ),
          ]),
        ],
      );

  Widget _pageModels(bool busy) => Padding(
        padding: const EdgeInsets.fromLTRB(20, 18, 20, 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(children: [
              const Text('Local Models',
                  style:
                      TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              const Spacer(),
              Text('${_models.length} found',
                  style: const TextStyle(color: Colors.white54)),
              const SizedBox(width: 8),
              IconButton(
                  onPressed: _refreshModels,
                  icon: const Icon(Icons.refresh)),
            ]),
            const SizedBox(height: 4),
            const Text('Scanned from HF cache, LM Studio, and lemonade dirs.',
                style: TextStyle(color: Colors.white70, fontSize: 13)),
            const SizedBox(height: 12),
            Expanded(child: _modelsList(busy)),
          ],
        ),
      );

  // ════════════════════════════ WIDGETS ════════════════════════════

  Widget _destRepoField() {
    final items = <DropdownMenuItem<String>>[
      for (final r in _repos) DropdownMenuItem(value: r, child: Text(r)),
      const DropdownMenuItem(
          value: '__new__',
          child: Text('➕ Create new repo…',
              style: TextStyle(fontStyle: FontStyle.italic))),
    ];
    final current = _creatingRepo
        ? '__new__'
        : (_repos.contains(_ulDest.text) ? _ulDest.text : null);
    return DropdownButtonFormField<String>(
      isDense: true,
      isExpanded: true,
      initialValue: current,
      decoration: const InputDecoration(
        isDense: true,
        labelText: 'Destination repo',
        border: OutlineInputBorder(),
      ),
      items: items,
      onChanged: (v) {
        setState(() {
          if (v == '__new__') {
            _creatingRepo = true;
            _syncNewRepo();
          } else {
            _creatingRepo = false;
            _ulDest.text = v ?? '';
          }
        });
      },
    );
  }

  Widget _newRepoRow() => Row(children: [
        if (_namespaces.isNotEmpty)
          SizedBox(
            width: 180,
            child: DropdownButtonFormField<String>(
              isDense: true,
              isExpanded: true,
              initialValue: _newRepoNs,
              decoration: const InputDecoration(
                isDense: true,
                labelText: 'Namespace',
                border: OutlineInputBorder(),
              ),
              items: [
                for (final ns in _namespaces)
                  DropdownMenuItem(value: ns, child: Text(ns)),
              ],
              onChanged: (v) => setState(() {
                _newRepoNs = v;
                _syncNewRepo();
              }),
            ),
          ),
        if (_namespaces.isNotEmpty) const SizedBox(width: 8),
        Expanded(
          child: TextField(
            controller: _newRepoName,
            onChanged: (_) => setState(_syncNewRepo),
            decoration: InputDecoration(
              isDense: true,
              labelText: _namespaces.isEmpty
                  ? 'New repo (namespace/name)'
                  : 'New repo name',
              border: const OutlineInputBorder(),
            ),
          ),
        ),
      ]);

  Widget _statusDot(bool ok, String yes, String no) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(ok ? Icons.check_circle : Icons.radio_button_unchecked,
              size: 18, color: ok ? Colors.greenAccent : Colors.white38),
          const SizedBox(width: 4),
          Text(ok ? yes : no, style: const TextStyle(fontSize: 13)),
        ],
      );

  Widget _kv(String k, String v) => RichText(
        text: TextSpan(
          style: const TextStyle(fontSize: 13),
          children: [
            TextSpan(text: '$k: ', style: const TextStyle(color: Colors.white54)),
            TextSpan(
                text: v,
                style: const TextStyle(
                    color: Colors.white, fontWeight: FontWeight.w600)),
          ],
        ),
      );

  Widget _btn(String label, IconData icon, VoidCallback? onTap) =>
      FilledButton.tonalIcon(
          onPressed: onTap, icon: Icon(icon, size: 18), label: Text(label));

  Widget _modelsList(bool busy) => Card(
        child: _models.isEmpty
            ? const Center(
                child: Text('No models found.\nDownload one in Step 2, '
                    'then Rescan.', textAlign: TextAlign.center))
            : ListView.separated(
                itemCount: _models.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, i) {
                  final m = _models[i];
                  return ListTile(
                    isThreeLine: true,
                    title: Tooltip(
                      message: '${m.name}\n${m.path}',
                      child: SelectableText(m.name,
                          style: const TextStyle(fontSize: 13)),
                    ),
                    subtitle: Text(
                        '${m.source} · ${m.format} · '
                        '${m.sizeGb.toStringAsFixed(1)} GB\n${m.path}',
                        style: const TextStyle(fontSize: 11),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis),
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.copy, size: 16),
                          tooltip: 'Copy path',
                          onPressed: () => Clipboard.setData(
                              ClipboardData(text: m.path)),
                        ),
                        IconButton(
                          icon: const Icon(Icons.delete_outline, size: 18),
                          tooltip: 'Delete from disk',
                          onPressed: busy ? null : () => _deleteModel(m),
                        ),
                      ],
                    ),
                  );
                },
              ),
      );

  // ── Persistent log dock (visible on every page) ──
  Widget _logDock() {
    return Container(
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: Colors.white12)),
        color: Colors.black,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          InkWell(
            onTap: () => setState(() => _logExpanded = !_logExpanded),
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              child: Row(children: [
                Icon(
                    _logExpanded
                        ? Icons.keyboard_arrow_down
                        : Icons.keyboard_arrow_up,
                    size: 18),
                const SizedBox(width: 6),
                const Text('Live log',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
                const SizedBox(width: 8),
                Text('${_log.length} lines',
                    style:
                        const TextStyle(fontSize: 11, color: Colors.white54)),
                const Spacer(),
                IconButton(
                  icon: const Icon(Icons.copy_all, size: 16),
                  tooltip: 'Copy log',
                  visualDensity: VisualDensity.compact,
                  onPressed: () =>
                      Clipboard.setData(ClipboardData(text: _log.join('\n'))),
                ),
                IconButton(
                  icon: const Icon(Icons.delete_sweep, size: 16),
                  tooltip: 'Clear log',
                  visualDensity: VisualDensity.compact,
                  onPressed: () => setState(_log.clear),
                ),
              ]),
            ),
          ),
          if (_logExpanded)
            SizedBox(
              height: 190,
              child: ListView.builder(
                controller: _logScroll,
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
                itemCount: _log.length,
                itemBuilder: (_, i) => SelectableText(
                  _log[i],
                  style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      color: Color(0xFFB9F6CA)),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
