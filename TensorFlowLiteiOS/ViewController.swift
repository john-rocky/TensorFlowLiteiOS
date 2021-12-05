//
//  ViewController.swift
//  TensorFlowLiteiOS
//
//  Created by 間嶋大輔 on 2021/12/05.
//

import UIKit
import AVFoundation
import TensorFlowLite


class ViewController: UIViewController,AVCaptureVideoDataOutputSampleBufferDelegate {
    
    
    private var interpreter:Interpreter!
    private var labels:[String] = []
    
    var modelName = "mobilenet_quant_v1_224"
    
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 224
    let inputHeight = 224
    
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var resultsLabel: UILabel!
    
    var captureSession = AVCaptureSession()
    var previewLayer:AVCaptureVideoPreviewLayer?
    var videoDataOutput = AVCaptureVideoDataOutput()
    
    var frameCount:Int = 0
    var resultCount:Int = 1
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupInterpreter()
        setupLabels()
        setupAV()
        setupResultsView()
        // Do any additional setup after loading the view.
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameCount += 1
        if frameCount >= 60 {
            frameCount = 0
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            runModel(pixelBuffer: pixelBuffer)
            
        }
    }
    
    
    func setupInterpreter() {
        
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "tflite") else { print("Failed to load the model."); return }
        
        var options = Interpreter.Options()
        options.threadCount = 1
        do {
            // Interpreter(通訳者)として初期化
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // 入力テンソルのためにメモリを割り当てる
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return
        }
    }
    
    func setupLabels() {
        guard let fileURL = Bundle.main.url(forResource: "labels", withExtension: "txt") else { fatalError("Labels file not found in bundle. Please add a labels.") }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            self.labels = contents.components(separatedBy: .newlines)
        } catch {
            fatalError("Labels file cannot be read.")
        }
    }
    
    func runModel(pixelBuffer: CVPixelBuffer) {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
            return
        }
        let outputTensor: Tensor
        do {
            let inputTensor = try interpreter.input(at: 0)

            guard let rgbData = rgbDataFromBuffer(
                thumbnailPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else { print("Failed to convert the image buffer to RGB data."); return }

            try interpreter.copy(rgbData, toInputAt: 0)

            try interpreter.invoke()

            outputTensor = try interpreter.output(at: 0)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)") ;return
        }
        let results: [Float]
        switch outputTensor.dataType {
        case .uInt8:
            guard let quantization = outputTensor.quantizationParameters else {
                print("No results returned because the quantization values for the output tensor are nil.")
                return
            }
            let quantizedResults = [UInt8](outputTensor.data)
            results = quantizedResults.map {
                quantization.scale * Float(Int($0) - quantization.zeroPoint)
            }
        case .float32:
            results = [Float32](unsafeData: outputTensor.data) ?? []
        default:
            print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
            return
        }
        getTopNLabels(results: results)
    }
    
    func getTopNLabels(results:[Float]) {
        // ラベル番号と信頼度のtupleの配列を作る [(labelIndex: Int, confidence: Float)]
        let zippedResults = zip(labels.indices, results)
        
        // 信頼度の高い順に並べ替え、resultCountの個数取得
        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)
        
        print(sortedResults)
        let label = labels[sortedResults[0].0]
        let confidence = floor(sortedResults[0].1*1000)/1000
        let resultText = modelName + "\n\(label)" + "\n\(confidence)"
        DispatchQueue.main.async {
            self.resultsLabel.text = resultText
        }
    }
    
    func setupAV() {
        let device = AVCaptureDevice.default(for: AVMediaType.video)
        let deviceInput = try! AVCaptureDeviceInput(device: device!)

        captureSession.addInput(deviceInput)
        
        videoDataOutput.videoSettings = [ String(kCVPixelBufferPixelFormatTypeKey) : kCMPixelFormat_32BGRA]
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoDataOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.frame = previewView.bounds
        previewLayer?.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
        previewView.layer.addSublayer(previewLayer!)

        captureSession.startRunning()
    }
    
    func setupResultsView() {
        resultsLabel.numberOfLines = 3
        resultsLabel.text = modelName
    }
}

