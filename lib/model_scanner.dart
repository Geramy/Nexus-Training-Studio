import 'dart:io';

/// Delete a scanned model from disk. For an HF-cache model it removes the whole
/// `models--org--name` dir (blobs + snapshots); for a GGUF it removes the file;
/// otherwise the model directory.
Future<void> deleteModel(ModelInfo m) async {
  final p = m.path;
  final snap = p.indexOf('/snapshots/');
  if (p.contains('/hub/models--') && snap >= 0) {
    final root = Directory(p.substring(0, snap));
    if (root.existsSync()) await root.delete(recursive: true);
    return;
  }
  if (m.format == 'gguf') {
    final f = File(p);
    if (f.existsSync()) await f.delete();
    return;
  }
  final d = Directory(p);
  if (d.existsSync()) await d.delete(recursive: true);
}

/// A model found on disk by [ModelScanner].
class ModelInfo {
  final String name;
  final String path;
  final String format; // 'safetensors' | 'gguf'
  final double sizeGb;
  final String source; // 'HuggingFace' | 'LM Studio' | 'lemonade' | 'extra'

  ModelInfo(this.name, this.path, this.format, this.sizeGb, this.source);

  Map<String, dynamic> toJson() => {
        'name': name,
        'path': path,
        'format': format,
        'sizeGb': double.parse(sizeGb.toStringAsFixed(2)),
        'source': source,
      };
}

/// Scans the usual local model caches: Hugging Face, LM Studio, lemonade, and any
/// extra dirs from config. A "model" = a folder with a config.json (safetensors)
/// or any *.gguf file. Bounded recursion so it stays fast.
class ModelScanner {
  final List<String> extraDirs;
  ModelScanner({this.extraDirs = const []});

  String get _home => Platform.environment['HOME'] ?? '';

  List<({String dir, String source})> get _roots => [
        (dir: '$_home/.cache/huggingface/hub', source: 'HuggingFace'),
        (dir: '$_home/.lmstudio/models', source: 'LM Studio'),
        (dir: '$_home/.cache/lm-studio/models', source: 'LM Studio'),
        (dir: '$_home/.cache/lemonade', source: 'lemonade'),
        for (final d in extraDirs) (dir: d, source: 'extra'),
      ];

  Future<List<ModelInfo>> scan() async {
    final found = <String, ModelInfo>{}; // by path, de-duped
    for (final root in _roots) {
      final dir = Directory(root.dir);
      if (!dir.existsSync()) continue;
      await _walk(dir, root.source, found, depth: 0);
    }
    final list = found.values.toList()
      ..sort((a, b) => a.name.toLowerCase().compareTo(b.name.toLowerCase()));
    return list;
  }

  Future<void> _walk(Directory dir, String source,
      Map<String, ModelInfo> out, {required int depth}) async {
    if (depth > 4) return;
    List<FileSystemEntity> entries;
    try {
      // followLinks: true — HF cache stores snapshot files (config.json, *.gguf)
      // as SYMLINKS into blobs/, so we must resolve them to be seen as files.
      entries = dir.listSync(followLinks: true);
    } catch (_) {
      return;
    }
    final files = entries.whereType<File>().toList();
    final hasConfig = files.any((f) => f.path.endsWith('/config.json'));
    final ggufs = files.where((f) => f.path.endsWith('.gguf')).toList();

    if (hasConfig) {
      out[dir.path] = ModelInfo(_pretty(dir.path), dir.path, 'safetensors',
          _dirSizeGb(dir), source);
    }
    for (final g in ggufs) {
      out[g.path] = ModelInfo(
          g.uri.pathSegments.last, g.path, 'gguf', _fileGb(g), source);
    }
    // Recurse into subdirectories (HF "hub" + LM Studio nest a couple levels).
    for (final sub in entries.whereType<Directory>()) {
      await _walk(sub, source, out, depth: depth + 1);
    }
  }

  String _pretty(String path) {
    // HF cache: the config.json lives in .../models--org--name/snapshots/<hash>,
    // so derive the name from the models--org--name ANCESTOR (not the hash dir).
    final m = RegExp(r'models--([^/]+)').firstMatch(path);
    if (m != null) return m.group(1)!.replaceAll('--', '/');
    return path.split('/').last;
  }

  double _fileGb(File f) {
    try {
      return f.lengthSync() / (1024 * 1024 * 1024);
    } catch (_) {
      return 0;
    }
  }

  double _dirSizeGb(Directory d) {
    var bytes = 0;
    try {
      for (final e in d.listSync(recursive: true, followLinks: false)) {
        if (e is File) {
          try {
            bytes += e.lengthSync();
          } catch (_) {}
        }
      }
    } catch (_) {}
    return bytes / (1024 * 1024 * 1024);
  }
}
