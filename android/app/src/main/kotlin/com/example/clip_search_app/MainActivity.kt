package com.example.clip_search_app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import ai.onnxruntime.*
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import kotlin.math.sqrt

class MainActivity : FlutterActivity() {
    private val CHANNEL = "clip.infer"

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler {
            call, result ->
            if (call.method == "runClip") {
                val imageBytes = call.argument<ByteArray>("image")
                val prompt = call.argument<String>("prompt")

                if (imageBytes == null || prompt == null) {
                    result.success("uncertain")
                    return@setMethodCallHandler
                }

                val classification = classifyImageWithPrompt(imageBytes, prompt)
                result.success(classification)
            }
        }
    }

    private fun classifyImageWithPrompt(imageBytes: ByteArray, prompt: String): String {
        val env = OrtEnvironment.getEnvironment()
        val sessionOptions = OrtSession.SessionOptions()

        val imageModel = loadModelFromAsset(env, "clip_image_encoder.onnx")
        val textModel = loadModelFromAsset(env, "clip_text_encoder.onnx")

        val imageTensor = preprocessImage(imageBytes, env)
        val promptTensor = preprocessPrompt(env)  // STATIC TOKEN for now

        val imageOutput = imageModel.run(mapOf("image" to imageTensor))
        val textOutput = textModel.run(mapOf("tokens" to promptTensor))

        val imageVec = (imageOutput[0].value as Array<FloatArray>)[0]
        val textVec = (textOutput[0].value as Array<FloatArray>)[0]

        val similarity = cosineSimilarity(imageVec, textVec)
println("📊 Cosine similarity score: $similarity")
        return when {
            similarity > 0.7 -> "approve"
            similarity > 0.5 -> "uncertain"
            else -> "reject"
        }
    }

    private fun loadModelFromAsset(env: OrtEnvironment, assetName: String): OrtSession {
        val file = File(cacheDir, assetName)
        if (!file.exists()) {
            val input = assets.open(assetName)
            val output = FileOutputStream(file)
            input.copyTo(output)
            input.close()
            output.close()
        }

        return env.createSession(file.absolutePath, OrtSession.SessionOptions())
    }

    private fun preprocessImage(imageBytes: ByteArray, env: OrtEnvironment): OnnxTensor {
        val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val floatValues = FloatArray(3 * 224 * 224)
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        var idx = 0
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resized.getPixel(x, y)
                val r = ((pixel shr 16) and 0xff) / 255.0f
                val g = ((pixel shr 8) and 0xff) / 255.0f
                val b = (pixel and 0xff) / 255.0f

                floatValues[idx++] = (r - mean[0]) / std[0]
                floatValues[idx++] = (g - mean[1]) / std[1]
                floatValues[idx++] = (b - mean[2]) / std[2]
            }
        }

        val shape = longArrayOf(1, 3, 224, 224)
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floatValues), shape)
    }

    private fun preprocessPrompt(env: OrtEnvironment): OnnxTensor {
        // Static tokenized prompt (example: 22 real tokens)
        val actualTokens = intArrayOf(
49406, 320, 13568, 2569, 593, 21426, 49407
        )

        // Pad to 77
        val padded = IntArray(77) { 0 }
        for (i in actualTokens.indices) {
            padded[i] = actualTokens[i]
        }

        val shape = longArrayOf(1, 77)
        val buffer = java.nio.IntBuffer.wrap(padded)
        return OnnxTensor.createTensor(env, buffer, shape)
    }


    private fun cosineSimilarity(vec1: FloatArray, vec2: FloatArray): Float {
        var dot = 0.0f
        var norm1 = 0.0f
        var norm2 = 0.0f
        for (i in vec1.indices) {
            dot += vec1[i] * vec2[i]
            norm1 += vec1[i] * vec1[i]
            norm2 += vec2[i] * vec2[i]
        }
        return dot / (sqrt(norm1) * sqrt(norm2))
    }
}
