import 'dart:convert';
import 'dart:io';

/// Runs the Python/MLX pipeline + Hugging Face model management as subprocesses,
/// streaming output to [onLog]. Flutter is only the control plane — MLX (native
/// C++/Metal under the hood) does the training. The app manages the Python venv
/// so the user never touches a terminal.
class Pipeline {
  final String studioRoot;
  final String configPython; // fallback interpreter if no venv yet
  final String baseModel;
  final String llamaCppDir;
  final String quant;
  final void Function(String line) onLog;

  Process? _current;
  bool get running => _current != null;

  Pipeline({
    required this.studioRoot,
    required this.configPython,
    required this.baseModel,
    required this.llamaCppDir,
    required this.quant,
    required this.onLog,
  });

  String get _venvBin => '$studioRoot/.venv/bin';
  bool get envReady => File('$_venvBin/python').existsSync();
  String get _python => envReady ? '$_venvBin/python' : configPython;
  String get _tokenPath => '$studioRoot/workspace/hf_token.txt';

  String? get hfToken {
    final f = File(_tokenPath);
    if (!f.existsSync()) return null;
    final t = f.readAsStringSync().trim();
    return t.isEmpty ? null : t;
  }

  void saveToken(String token) {
    final f = File(_tokenPath)..parent.createSync(recursive: true);
    f.writeAsStringSync(token.trim());
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
        ['-lc',
            '$configPython -m venv .venv && .venv/bin/pip install -U pip -q && '
            '.venv/bin/pip install -q -r py/requirements.txt && echo ENV_READY'],
        'Setup Python env (venv + mlx-lm)',
      );

  // ── Hugging Face ───────────────────────────────────────────────────────
  Future<int> downloadModel(String repo) =>
      _run(_python, ['py/hf_download.py', repo], 'Download $repo');

  Future<int> uploadModel(String localPath, String destRepo, bool private) =>
      _run(_python, [
        'py/hf_upload.py', localPath, destRepo, if (private) '--private',
      ], 'Upload → $destRepo');

  // ── Training pipeline ──────────────────────────────────────────────────
  Future<int> checkSupport() => _run(
      _python, ['py/inspect_model.py', baseModel], 'Check support (inspect model)');

  Future<int> prepareData() =>
      _run(_python, ['py/prepare_data.py'], 'Prepare data');

  Future<int> train() => _run(
      'bash', ['py/train.sh', baseModel, 'py/lora_config.yaml'], 'Train (LoRA)');

  Future<int> evaluate() =>
      _run(_python, ['py/eval.py', '--model', 'workspace/fused'], 'Eval (fused)');

  Future<int> exportGguf() => _run(
      'bash', ['py/export_gguf.sh', baseModel, llamaCppDir, quant], 'Export GGUF');
}
