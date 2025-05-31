import 'dart:developer';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'Clip QC',
      home: ClipQCPage(),
    );
  }
}

class ClipQCPage extends StatefulWidget {
  const ClipQCPage({super.key});

  @override
  State<ClipQCPage> createState() => _ClipQCPageState();
}

class _ClipQCPageState extends State<ClipQCPage> {
  final MethodChannel platform = const MethodChannel("clip.infer");
  Uint8List? selectedImage;
  String result = "";
  bool loading = false;

  static const String staticPrompt = '''
You are an image evaluator for quality control. Your task is to analyze images of repair work for defects and provide a one-word response based on the following criteria:

approve: The image clearly shows a defect, even if it is minor.

reject: The image does not show any defect and every object in image is clear and not have any sinlge spot of damage.

uncertain: The image is unclear, and it is difficult to determine whether to approve or reject.

Please return only one-word response ('approve', 'reject', or 'uncertain')
''';

  Future<void> pickImageAndClassify() async {
    final picker = ImagePicker();
    final file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    final imageBytes = await file.readAsBytes();

    setState(() {
      selectedImage = imageBytes;
      result = "";
      loading = true;
    });

    try {
      final response = await platform.invokeMethod<String>("runClip", {
        "image": imageBytes,
        "prompt": staticPrompt,
      });
      inspect(response);
      setState(() {
        result = response ?? "uncertain";
        loading = false;
      });
    } catch (e) {
      setState(() {
        result = "Error: $e";
        loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Offline QC Classifier")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: pickImageAndClassify,
              child: const Text("Pick Image"),
            ),
            const SizedBox(height: 16),
            if (selectedImage != null)
              Image.memory(selectedImage!, height: 200),
            const SizedBox(height: 16),
            if (loading)
              const CircularProgressIndicator()
            else if (result.isNotEmpty)
              Text("Result: $result", style: const TextStyle(fontSize: 20)),
          ],
        ),
      ),
    );
  }
}
