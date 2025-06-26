import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // 1. Minta dan periksa status izin kamera
  final cameraPermissionStatus = await Permission.camera.request();
  // Mendapatkan daftar kamera yang tersedia
  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(MyApp(camera: firstCamera));

  // 2. Lakukan pengecekan SEBELUM mencoba menggunakan kamera
  if (cameraPermissionStatus.isGranted) {
    // Izin diberikan, SEKARANG baru aman untuk mencari kamera
    final cameras = await availableCameras();

    // 3. Periksa lagi apakah daftar kamera tidak kosong
    if (cameras.isNotEmpty) {
      // Jika ada kamera, jalankan aplikasi utama
      runApp(MyApp(camera: cameras.first));
    } else {
      // Jika izin diberikan tapi tidak ada kamera, tampilkan error
      runApp(const ErrorApp("Tidak ada kamera yang ditemukan di perangkat ini."));
    }
  } else {
    // Jika izin ditolak, tampilkan error
    runApp(const ErrorApp("Izin kamera ditolak. Aplikasi tidak dapat berjalan."));
  }
}

class ErrorApp extends StatelessWidget {
  final String errorMessage;
  const ErrorApp(this.errorMessage, {super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text("Error"),
          backgroundColor: Colors.red,
        ),
        body: Center(
          child: Text(errorMessage),
        ),
      ),
    );
  }
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  //const MyApp({Key? key, required this.camera}) : super(key: key);
  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLOv8 Detection',
      theme: ThemeData.dark(),
      home: ObjectDetectionView(camera: camera),
    );
  }
}

class ObjectDetectionView extends StatefulWidget {
  final CameraDescription camera;
  //const ObjectDetectionView({Key? key, required this.camera}) : super(key: key);
  const ObjectDetectionView({super.key, required this.camera});

  @override
  //_ObjectDetectionViewState createState() => _ObjectDetectionViewState();
  State<ObjectDetectionView> createState() => _ObjectDetectionViewState();
}

class _ObjectDetectionViewState extends State<ObjectDetectionView> {
  late CameraController _cameraController;
  late Interpreter _interpreter;
  late List<String> _labels;
  bool _isDetecting = false;
  List<dynamic>? _recognitions;
  int _imageHeight = 0;
  int _imageWidth = 0;

  @override
  void initState() {
    super.initState();
    _cameraController = CameraController(widget.camera, ResolutionPreset.high);
    _cameraController.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});
      // Mulai stream gambar dari kamera
      _cameraController.startImageStream((CameraImage image) {
        if (!_isDetecting) {
          _isDetecting = true;
          _runModelOnFrame(image);
        }
      });
    });

    _loadModel();
  }

  Future<void> _loadModel() async {
    // Memuat model TFLite dari assets
    _interpreter = await Interpreter.fromAsset('assets/best.tflite');
    // Memuat label dari assets
    if (!mounted) return;
    //final labelsData = await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
    final labelsData = await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
    _labels = labelsData.split('\n');
  }

  Future<void> _runModelOnFrame(CameraImage cameraImage) async {
    // Ukuran input model YOLOv8 (misal: 640x640)
    const int modelInputSize = 640;
    // Ambang batas kepercayaan untuk menampilkan deteksi
    const double confidenceThreshold = 0.5;

    // 1. Pre-process gambar dari kamera
    final inputImage = _preprocessCameraImage(cameraImage, modelInputSize);

    // 2. Siapkan input dan output tensor untuk model
    // Bentuk output YOLOv8 biasanya [1, 84, 8400] atau [1, 7, 8400]
    // 7 = 4 (box) + 3 (class scores)
    final outputShape = _interpreter.getOutputTensor(0).shape;
    final output = List.filled(outputShape.reduce((a, b) => a * b), 0.0).reshape(outputShape);

    // 3. Jalankan inferensi
    _interpreter.run(inputImage, output);

    // 4. Post-process hasil output
    List<Map<String, dynamic>> recognitions = [];
    for (int i = 0; i < output[0][0].length; i++) {
        // Output[0] -> [x, y, w, h, class1_score, class2_score, class3_score]
        final scores = output[0].sublist(4);
        var maxScore = 0.0;
        var bestClassIndex = -1;

        // Cari kelas dengan skor tertinggi
        for (int j = 0; j < scores.length; j++) {
            if (scores[j] > maxScore) {
                maxScore = scores[j];
                bestClassIndex = j;
            }
        }
        
        // Filter berdasarkan confidence threshold
        if (maxScore > confidenceThreshold) {
            final box = output[0].sublist(0, 4);

            // Konversi koordinat dari 0-1 ke ukuran layar
            recognitions.add({
                "rect": Rect.fromCenter(
                  center: Offset(box[0] * _imageWidth, box[1] * _imageHeight),
                  width: box[2] * _imageWidth,
                  height: box[3] * _imageHeight,
                ),
                "detectedClass": _labels[bestClassIndex],
                "confidenceInClass": maxScore,
            });
        }
    }


    // Update UI
    setState(() {
      _recognitions = recognitions;
      _imageHeight = cameraImage.height;
      _imageWidth = cameraImage.width;
    });

    _isDetecting = false;
  }

  // Fungsi untuk pre-processing gambar (penting!)
  img.Image _preprocessCameraImage(CameraImage image, int inputSize) {
    // Konversi dari format YUV420 ke gambar RGB
    img.Image? convertedImage;

    if (image.format.group == ImageFormatGroup.yuv420) {
      convertedImage = _convertYUV420toImage(image);
    } else if (image.format.group == ImageFormatGroup.bgra8888) {
      convertedImage = img.Image.fromBytes(
        width: image.width,
        height: image.height,
        bytes: image.planes[0].bytes.buffer,
        order: img.ChannelOrder.bgra,
      );
    }

    if (convertedImage == null) return img.Image(width: 0, height: 0);

    // Resize gambar ke ukuran input model (640x640)
    final resizedImage = img.copyResize(convertedImage, width: inputSize, height: inputSize);
    return resizedImage;
  }
  
  // Konversi YUV_420_888 ke Image
  // (Fungsi helper, bisa dicari implementasi lengkapnya online)
  img.Image _convertYUV420toImage(CameraImage cameraImage) {
    // ... implementasi konversi format gambar ...
    // Ini bagian yang cukup teknis, Anda bisa menemukan Gist atau package untuk ini.
    // Secara sederhana, ini mengubah format data mentah dari kamera ke format RGB standar.
    // Untuk sementara kita return gambar kosong agar kode bisa berjalan
    return img.Image(width: cameraImage.width, height: cameraImage.height);
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return Container();
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Deteksi Fraksi Buah')),
      // ... di dalam method build
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          // Bungkus BoundingBoxPainter dengan CustomPaint
          CustomPaint(
            painter: BoundingBoxPainter(
              recognitions: _recognitions ?? [],
              previewHeight: MediaQuery.of(context).size.height,
              previewWidth: MediaQuery.of(context).size.width,
              screenHeight: _imageHeight.toDouble(),
              screenWidth: _imageWidth.toDouble(),
            ),
          ),
        ],
      ),
      // ...
    );
  }
}


// Kelas Painter untuk menggambar Bounding Box
class BoundingBoxPainter extends CustomPainter {
  final List<dynamic> recognitions;
  final double previewHeight;
  final double previewWidth;
  final double screenHeight;
  final double screenWidth;

  BoundingBoxPainter({
    required this.recognitions,
    required this.previewHeight,
    required this.previewWidth,
    required this.screenHeight,
    required this.screenWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (recognitions.isEmpty) return;

    final scaleX = size.width / screenWidth;
    final scaleY = size.height / screenHeight;

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = Colors.red;

    for (var recognition in recognitions) {
      final rect = recognition['rect'] as Rect;
      final scaledRect = Rect.fromLTRB(
        rect.left * scaleX,
        rect.top * scaleY,
        rect.right * scaleX,
        rect.bottom * scaleY,
      );

      canvas.drawRect(scaledRect, paint);

      TextSpan span = TextSpan(
        text: '${recognition['detectedClass']} ${(recognition['confidenceInClass'] * 100).toStringAsFixed(0)}%',
        style: const TextStyle(color: Colors.white, fontSize: 12),
      );
      TextPainter tp = TextPainter(text: span, textAlign: TextAlign.left, textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(scaledRect.left, scaledRect.top - 20));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}