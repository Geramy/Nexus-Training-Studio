import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// Runs the Python/MLX pipeline + Hugging Face model management as subprocesses,
/// streaming output to [onLog]. Flutter is only the control plane — MLX (native
/// C++/Metal under the hood) does the training. The app manages the Python venv
/// so the user never touches a terminal.
class Pipeline {
  final String studioRoot;
  final String configPython; // fallback interpreter if no venv yet
  final String baseModel;
  final String llamaCppDir;
  final int trainBits; // QLoRA base precision (8)
  final String exportQuants; // CSV, e.g. "Q8_0,Q6_K,Q4_K_M"
  final void Function(String line) onLog;

  Process? _current;
  bool get running => _current != null;

  Pipeline({
    required this.studioRoot,
    required this.configPython,
    required this.baseModel,
    required this.llamaCppDir,
    required this.trainBits,
    required this.exportQuants,
    required this.onLog,
  });

  String get _venvBin => '$studioRoot/.venv/bin';
  bool get envReady => File('$_venvBin/python').existsSync();
  String get _python => envReady ? '$_venvBin/python' : configPython;
  String get _tokenPath => '$studioRoot/workspace/hf_token.txt';

  /// The 8-bit MLX base built by [quantizeBase]; training/fuse use it when present.
  String get base8bitPath => '$studioRoot/workspace/base-8bit';
  bool get base8bitReady => File('$base8bitPath/config.json').existsSync();
  String get trainBase => base8bitReady ? base8bitPath : baseModel;

  String? get hfToken {
    final f = File(_tokenPath);
    if (!f.existsSync()) return null;
    final t = f.readAsStringSync().trim();
    return t.isEmpty ? null : t;
  }

  void saveToken(String token) {
    File(_tokenPath)
      ..parent.createSync(recursive: true)
      ..writeAsStringSync(token.trim());
    onLog(token.trim().isEmpty ? 'HF token cleared.' : 'HF token saved.');
  }

  Map<String, String> get _env {
    final t = hfToken;
    return {
      ...Platform.environment,
      'PATH': '$_venvBin:${Platform.environment['PATH'] ?? ''}',
      if (t != null) 'HF_TOKEN': t,
      if (t != null) 'HUGGING_FACE_HUB_TOKEN': t,
    };
  }

  Future<int> _run(String exe, List<String> args, String label) async {
    if (running) {
      onLog('⚠ a step is already running — wait for it to finish.');
      return -1;
    }
    onLog('\n▶ $label\n  \$ $exe ${args.join(' ')}');
    try {
      final p = await Process.start(exe, args,
          workingDirectory: studioRoot, environment: _env, runInShell: false);
      _current = p;
      p.stdout
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(onLog);
      p.stderr
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(onLog);
      final code = await p.exitCode;
      _current = null;
      onLog(code == 0 ? '✓ $label done' : '✗ $label exited with $code');
      return code;
    } catch (e) {
      _current = null;
      onLog('✗ $label failed to start: $e');
      return -1;
    }
  }

  void cancel() {
    _current?.kill(ProcessSignal.sigterm);
    _current = null;
    onLog('■ cancelled');
  }

  // ── Environment ────────────────────────────────────────────────────────
  Future<int> setupEnv() => _run(
        'bash',
        [
          '-lc',
          '$configPython -m venv .venv && .venv/bin/pip install -U pip -q && '
              '.venv/bin/pip install -q -r py/requirements.txt && echo ENV_READY',
        ],
        'Setup Python env (venv + mlx-lm)',
      );

  // ── Hugging Face ───────────────────────────────────────────────────────
  Future<int> downloadModel(String repo) =>
      _run(_python, ['py/hf_download.py', repo], 'Download $repo');

  Future<int> uploadModel(String localPath, String destRepo, bool private) =>
      _run(_python, [
        'py/hf_upload.py', localPath, destRepo, if (private) '--private',
      ], 'Upload → $destRepo');

  /// Search HF for model repos by free text (for the download-field autocomplete).
  Future<List<String>> searchModels(String query) async {
    if (query.trim().isEmpty) return [];
    try {
      final r = await Process.run(_python, ['py/hf_search.py', query, '25'],
          workingDirectory: studioRoot, environment: _env, runInShell: false);
      final out = '${r.stdout}'.trim();
      if (out.isEmpty) return [];
      final d = jsonDecode(out);
      if (d is List) return [for (final e in d) e.toString()];
      return [];
    } catch (_) {
      return [];
    }
  }

  /// Search HF for DATASET repos by free text (for the dataset-import autocomplete).
  Future<List<String>> searchDatasets(String query) async {
    if (query.trim().isEmpty) return [];
    try {
      final r = await Process.run(
          _python, ['py/hf_search.py', query, '25', '--datasets'],
          workingDirectory: studioRoot, environment: _env, runInShell: false);
      final out = '${r.stdout}'.trim();
      if (out.isEmpty) return [];
      final d = jsonDecode(out);
      if (d is List) return [for (final e in d) e.toString()];
      return [];
    } catch (_) {
      return [];
    }
  }

  /// List the HF repos/namespaces the saved token can write to (for the picker).
  /// Captures JSON instead of streaming to the log. Returns {} on any failure.
  Future<Map<String, dynamic>> listRepos() async {
    try {
      final r = await Process.run(_python, ['py/hf_repos.py'],
          workingDirectory: studioRoot, environment: _env, runInShell: false);
      final out = '${r.stdout}'.trim();
      if (out.isEmpty) return {'error': '${r.stderr}'.trim()};
      final d = jsonDecode(out);
      return d is Map<String, dynamic> ? d : {};
    } catch (e) {
      return {'error': '$e'};
    }
  }

  // ── Training pipeline ──────────────────────────────────────────────────
  Future<int> checkSupport() => _run(
      _python, ['py/inspect_model.py', baseModel], 'Check support (inspect model)');

  Future<int> prepareData() =>
      _run(_python, ['py/prepare_data.py'], 'Prepare data');

  /// Make the 8-bit MLX base (from [baseModel]) that QLoRA trains on.
  Future<int> quantizeBase() => _run(
        _python,
        ['-m', 'mlx_lm', 'convert', '--hf-path', baseModel, '-q', '--q-bits',
            '$trainBits', '--mlx-path', base8bitPath],
        'Quantize base → ${trainBits}-bit MLX',
      );

  /// LoRA train. [model] overrides the base; [iters] overrides config;
  /// [defaultKeys] runs a quick smoke with mlx-lm's auto LoRA targets (no config),
  /// e.g. to validate the toolchain on a small model.
  Future<int> train({String? model, int? iters, bool defaultKeys = false}) {
    final m = model ?? trainBase;
    if (defaultKeys) {
      return _run(_python, [
        '-m', 'mlx_lm', 'lora', '--model', m, '--train',
        '--data', 'workspace/data', '--adapter-path', 'workspace/adapters',
        '--iters', '${iters ?? 2}', '--num-layers', '2', '--batch-size', '1',
      ], 'Train (smoke, auto keys) on $m');
    }
    return _run('bash', [
      'py/train.sh', m, 'py/lora_config.yaml', if (iters != null) '$iters',
    ], 'Train (LoRA, ${trainBits}-bit base)');
  }

  Future<int> evaluate() =>
      _run(_python, ['py/eval.py', '--model', 'workspace/fused'], 'Eval (fused)');

  /// BFCL-style tool-call eval. Defaults to the TRAINED model = 8-bit base +
  /// the LoRA adapter (no fuse needed). Writes workspace/eval_result.json.
  Future<int> evalToolCalls(
      {String? model, String? adapter, int limit = 120, bool baseOnly = false}) {
    final m = model ?? (base8bitReady ? base8bitPath : baseModel);
    final a = adapter ?? '$studioRoot/workspace/adapters';
    final useAdapter =
        !baseOnly && File('$a/adapters.safetensors').existsSync();
    return _run(_python, [
      'py/eval_toolcalls.py', '--model', m,
      if (useAdapter) ...['--adapter', a],
      '--limit', '$limit',
    ], 'Tool-call eval (${useAdapter ? "base + adapter" : "base only"})');
  }

  /// The last eval's percentages, for the UI panel (null if none yet).
  Map<String, dynamic>? lastEvalResult() {
    final f = File('$studioRoot/workspace/eval_result.json');
    if (!f.existsSync()) return null;
    try {
      final d = jsonDecode(f.readAsStringSync());
      return d is Map<String, dynamic> ? d : null;
    } catch (_) {
      return null;
    }
  }

  /// END-TO-END interview eval against a SERVED GGUF endpoint (OpenAI-compatible —
  /// llama-server --jinja or lemonade). Simulates full setup interviews and scores
  /// COVERAGE (every required topic tagged), ONCE-ONLY (no topic re-asked), and
  /// COMPLETION, plus records TPS / PP-s from the server timings. Writes
  /// workspace/interview_result.json and per-run transcripts under
  /// workspace/interview_runs. Test cases live in the editable interview_cases seed.
  Future<int> evalInterview(
          {required String endpoint, String model = 'gguf', int scenarios = 8,
          String label = 'trained', int? caseIndex}) =>
      _run(_python, [
        'py/eval_interview.py', '--endpoint', endpoint,
        '--model', model, '--scenarios', '$scenarios', '--label', label,
        if (caseIndex != null) ...['--case', '$caseIndex'],
      ], 'Interview eval ($label${caseIndex != null ? ' #$caseIndex' : ''})');

  /// The last interview eval's metrics for the UI panel (null if none yet).
  Map<String, dynamic>? lastInterviewResult() {
    final f = File('$studioRoot/workspace/interview_result.json');
    if (!f.existsSync()) return null;
    try {
      final d = jsonDecode(f.readAsStringSync());
      return d is Map<String, dynamic> ? d : null;
    } catch (_) {
      return null;
    }
  }

  // ── Served-GGUF model servers for the interview A/B ───────────────────────
  // Long-running llama-server processes the studio starts/stops so the user can
  // serve the ORIGINAL ('base') and TRAINED ('trained') GGUFs and compare them.
  // Kept OUT of the single-step `_current` lane so a server never blocks pipeline
  // steps and two servers can run at once (base + trained on different ports).
  final Map<String, Process> _servers = {};

  bool serverRunning(String label) => _servers.containsKey(label);
  int serverPort(String label) => label == 'base' ? 8098 : 8099;
  String endpointFor(String label) =>
      'http://127.0.0.1:${serverPort(label)}/v1/chat/completions';

  /// Start a llama-server serving [gguf] on the port for [label] (base|trained),
  /// with tool-calling (--jinja) and the recommended serving setup (thinking-on
  /// via the template; the eval sets temperature per request). Streams the
  /// server's log into the studio log so you can watch the model load.
  Future<bool> startServer(
      {required String label, required String gguf, int contextSize = 16384}) async {
    if (_servers.containsKey(label)) {
      onLog('⚠ $label model server already running on :${serverPort(label)}.');
      return true;
    }
    // Accept an absolute path or one relative to the studio root (e.g. the
    // exported workspace/gguf/*.gguf).
    final path = gguf.startsWith('/') ? gguf : '$studioRoot/$gguf';
    if (!File(path).existsSync()) {
      onLog('✗ $label model not found: $path');
      return false;
    }
    gguf = path;
    final bin = File('$llamaCppDir/build/bin/llama-server').existsSync()
        ? '$llamaCppDir/build/bin/llama-server'
        : '$llamaCppDir/llama-server';
    final port = serverPort(label);
    final args = [
      '-m', gguf, '--jinja', '--host', '127.0.0.1', '--port', '$port',
      '-c', '$contextSize', '-ngl', '999',
    ];
    onLog('\n▶ start $label model (:$port)\n  \$ $bin ${args.join(' ')}');
    try {
      final p = await Process.start(bin, args,
          workingDirectory: studioRoot, environment: _env, runInShell: false);
      _servers[label] = p;
      void pipe(Stream<List<int>> s) => s
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen((l) => onLog('[$label] $l'));
      pipe(p.stdout);
      pipe(p.stderr);
      p.exitCode.then((c) {
        _servers.remove(label);
        onLog('• $label server exited ($c)');
      });
      onLog('✓ $label server starting on :$port — wait for "model loaded".');
      return true;
    } catch (e) {
      onLog('✗ $label server failed to start: $e');
      return false;
    }
  }

  void stopServer(String label) {
    final p = _servers.remove(label);
    if (p == null) {
      onLog('• no $label server running.');
      return;
    }
    p.kill(ProcessSignal.sigterm);
    onLog('• stopped $label server (:${serverPort(label)}).');
  }

  void stopAllServers() {
    for (final e in _servers.entries) {
      e.value.kill(ProcessSignal.sigterm);
    }
    _servers.clear();
  }

  /// Run the fine-tuned model against the use/test cases in workspace/tests and
  /// report pass/fail per case. [model] defaults to the fused fine-tune.
  Future<int> runTests({String? model}) => _run(
        _python,
        ['py/run_tests.py', '--model', model ?? 'workspace/fused'],
        'Test cases on ${model ?? "workspace/fused"}',
      );

  /// Fuse adapter (de-quantized) → GGUF → quantize to each of [exportQuants].
  Future<int> exportGguf() => _run('bash',
      ['py/export_gguf.sh', trainBase, llamaCppDir, exportQuants], 'Export GGUF');

  // ── Data management ──────────────────────────────────────────────────────
  /// Synthesize the setup/discovery/task corpus (full+partial+recovery variants).
  Future<int> generateData({String kinds = 'setup,discovery,tasks'}) => _run(
      _python, ['py/gen_training_data.py', '--kinds', kinds],
      'Generate training data ($kinds)');

  /// Import an Excel/CSV file of conversations into the dataset.
  Future<int> importExcel(String path) =>
      _run(_python, ['py/import_excel.py', path], 'Import Excel: $path');

  /// Write a fill-in Excel template matching our structure.
  Future<int> exportTemplate(String path) => _run(
      _python, ['py/import_excel.py', '--template', path], 'Write template: $path');

  /// Import a Hugging Face dataset, auto-mapping common schemas.
  Future<int> importHf(String dataset,
          {String split = 'train', String? config, int? limit}) =>
      _run(_python, [
        'py/import_hf_dataset.py', dataset, '--split', split,
        if (config != null && config.isNotEmpty) ...['--config', config],
        if (limit != null && limit > 0) ...['--limit', '$limit'],
      ], 'Import HF dataset: $dataset');

  // ── Seed data (editable JSON) — full in workspace/seeds, examples in seeds/ ─
  String get _seedsDir => '$studioRoot/workspace/seeds';
  String get _seedExamplesDir => '$studioRoot/seeds';

  List<String> seedNames() {
    final names = <String>{};
    final w = Directory(_seedsDir);
    if (w.existsSync()) {
      for (final f in w.listSync()) {
        if (f.path.endsWith('.json')) {
          names.add(f.uri.pathSegments.last.replaceAll('.json', ''));
        }
      }
    }
    final e = Directory(_seedExamplesDir);
    if (e.existsSync()) {
      for (final f in e.listSync()) {
        if (f.path.endsWith('.example.json')) {
          names.add(f.uri.pathSegments.last.replaceAll('.example.json', ''));
        }
      }
    }
    return names.toList()..sort();
  }

  /// Read a seed's editable JSON; bootstraps the workspace copy from the example.
  String readSeed(String name) {
    final w = File('$_seedsDir/$name.json');
    if (!w.existsSync()) {
      final ex = File('$_seedExamplesDir/$name.example.json');
      if (ex.existsSync()) {
        w.parent.createSync(recursive: true);
        w.writeAsStringSync(ex.readAsStringSync());
      } else {
        return '';
      }
    }
    return w.readAsStringSync();
  }

  /// Validate + save a seed's JSON. Returns null on success, else an error.
  String? writeSeed(String name, String content) {
    try {
      jsonDecode(content);
    } catch (e) {
      return 'Invalid JSON: $e';
    }
    File('$_seedsDir/$name.json')
      ..parent.createSync(recursive: true)
      ..writeAsStringSync(content);
    onLog('✓ seed "$name" saved');
    return null;
  }

  // ── Canonical dataset (dataset.jsonl) CRUD — the UI table edits this ───────
  String get datasetPath => '$studioRoot/workspace/data/dataset.jsonl';

  List<Map<String, dynamic>> datasetRows() {
    final f = File(datasetPath);
    if (!f.existsSync()) return [];
    final out = <Map<String, dynamic>>[];
    for (final line in f.readAsLinesSync()) {
      if (line.trim().isEmpty) continue;
      try {
        final d = jsonDecode(line);
        if (d is Map<String, dynamic>) out.add(d);
      } catch (_) {}
    }
    return out;
  }

  /// Memory-bounded count of rows — scans the file for newline bytes in chunks
  /// (the corpus can be multi-GB; never load it all). Caches the result in a
  /// sidecar so repeat loads are instant; the cache auto-invalidates when the
  /// dataset file is newer than the sidecar.
  int datasetCount() {
    final f = File(datasetPath);
    if (!f.existsSync()) return 0;
    final cache = File('$datasetPath.count');
    if (cache.existsSync() &&
        cache.lastModifiedSync().isAfter(f.lastModifiedSync())) {
      final v = int.tryParse(cache.readAsStringSync().trim());
      if (v != null) return v;
    }
    final n = _scanCount(f);
    try {
      cache.writeAsStringSync('$n');
    } catch (_) {}
    return n;
  }

  int _scanCount(File f) {
    final raf = f.openSync();
    try {
      var count = 0;
      const chunk = 1 << 20; // 1 MB
      var trailing = false; // last byte seen was non-newline
      while (true) {
        final bytes = raf.readSync(chunk);
        if (bytes.isEmpty) break;
        for (final b in bytes) {
          if (b == 10) {
            count++;
            trailing = false;
          } else {
            trailing = true;
          }
        }
      }
      return count + (trailing ? 1 : 0); // count a final unterminated line
    } finally {
      raf.closeSync();
    }
  }

  /// Compact per-row summaries for the table view — reads ONLY the prefix bytes
  /// up to (offset+limit) lines, so a multi-GB corpus never loads fully.
  List<Map<String, dynamic>> datasetSummary({int limit = 500, int offset = 0}) {
    final out = <Map<String, dynamic>>[];
    final f = File(datasetPath);
    if (!f.existsSync()) return out;
    final raf = f.openSync();
    try {
      final need = offset + limit;
      final buf = BytesBuilder(copy: false);
      var lines = 0;
      const chunk = 1 << 20;
      outer:
      while (lines < need) {
        final bytes = raf.readSync(chunk);
        if (bytes.isEmpty) break;
        buf.add(bytes);
        for (final b in bytes) {
          if (b == 10) {
            lines++;
            if (lines >= need) break outer;
          }
        }
      }
      final text = utf8.decode(buf.takeBytes(), allowMalformed: true);
      var i = -1;
      for (final line in const LineSplitter().convert(text)) {
        if (line.trim().isEmpty) continue;
        i++;
        if (i < offset) continue;
        if (out.length >= limit) break;
        try {
          final d = jsonDecode(line);
          if (d is Map<String, dynamic>) out.add(_summarize(d));
        } catch (_) {}
      }
    } finally {
      raf.closeSync();
    }
    return out;
  }

  Map<String, dynamic> _summarize(Map<String, dynamic> r) {
    final msgs = (r['messages'] as List?) ?? const [];
    var sys = '', firstUser = '';
    final calls = <String>{};
    for (final m in msgs) {
      if (m is! Map) continue;
      if (m['role'] == 'system' && sys.isEmpty) sys = '${m['content'] ?? ''}';
      if (m['role'] == 'user' && firstUser.isEmpty) {
        firstUser = '${m['content'] ?? ''}';
      }
      if (m['tool_calls'] is List) {
        for (final t in (m['tool_calls'] as List)) {
          if (t is Map && t['function'] is Map) {
            calls.add('${(t['function'] as Map)['name']}');
          }
        }
      }
    }
    var kind = 'other';
    if (sys.contains('Setup host')) {
      kind = 'setup';
    } else if (sys.contains('DISCOVERY')) {
      kind = 'discovery';
    } else if (sys.contains('Project Manager')) {
      kind = 'tasks';
    }
    return {
      'id': r['id'] ?? '',
      'source': r['source'] ?? '',
      'kind': kind,
      'turns': msgs.length,
      'tools': (r['tools'] as List?)?.length ?? 0,
      'calls': calls.toList(),
      'preview':
          firstUser.length > 140 ? '${firstUser.substring(0, 140)}…' : firstUser,
    };
  }

  /// FNV-1a 64-bit hex of the messages — stable id + dedupe for manual adds.
  String _hashMessages(Object messages) {
    final bytes = utf8.encode(jsonEncode(messages));
    var h = 0xcbf29ce484222325;
    const mask = 0xFFFFFFFFFFFFFFFF;
    for (final b in bytes) {
      h = (h ^ b) & mask;
      h = (h * 0x100000001b3) & mask;
    }
    return h.toRadixString(16).padLeft(16, '0');
  }

  /// Add a conversation row; returns false if invalid or a duplicate.
  bool addConversation(List<dynamic> messages,
      {String source = 'manual', List<dynamic>? tools}) {
    if (messages.isEmpty) return false;
    if (!messages.any((m) => m is Map && m['role'] == 'assistant')) return false;
    final id = _hashMessages(messages);
    final rows = datasetRows();
    if (rows.any((r) => r['id'] == id)) return false;
    rows.add({
      'id': id,
      'source': source,
      'messages': messages,
      if (tools != null && tools.isNotEmpty) 'tools': tools,
    });
    _writeDataset(rows);
    return true;
  }

  bool deleteConversation(String id) {
    final rows = datasetRows();
    final before = rows.length;
    rows.removeWhere((r) => r['id'] == id);
    if (rows.length == before) return false;
    _writeDataset(rows);
    return true;
  }

  void _writeDataset(List<Map<String, dynamic>> rows) {
    final f = File(datasetPath)..parent.createSync(recursive: true);
    f.writeAsStringSync(rows.isEmpty
        ? ''
        : '${rows.map(jsonEncode).join('\n')}\n');
  }
}
