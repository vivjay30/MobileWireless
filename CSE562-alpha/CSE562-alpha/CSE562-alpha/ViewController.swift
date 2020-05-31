//
//  ViewController.swift
//  CSE562-alpha
//
//  Created by Vivek Jayaram on 5/27/20.
//  Copyright © 2020 Vivek Jayaram. All rights reserved.
//

import UIKit
import AVKit

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    @IBOutlet weak var outputLabel: UILabel!
    
    var pixelIntensities: [Double]!
    let seqLength = 24  // Number of bits in our transmitted sequence
    let windowSize = 12  // Frames for each bit
    var transmittedText: String = ""
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
    
    @IBAction func startCapturing(_ sender: Any) {
        pixelIntensities = []

        // Video code based on a tutorial https://www.youtube.com/watch?v=p6GA8ODlnX0
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo

        guard let captureDevice = AVCaptureDevice.default(for: .video) else {return}
        configureCameraForHighestFrameRate(device: captureDevice)

        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else {return}
        captureSession.addInput(input)

        captureSession.startRunning()

        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)

        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame

        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(dataOutput)
        
        Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { (timer) in
            if (self.pixelIntensities.count >= 350){
                timer.invalidate()
                captureSession.stopRunning()
                previewLayer.removeFromSuperlayer()
                self.outputLabel.text = "Transmitted String: " + self.transmittedText
                return
            }
        }
    }
    

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let x = meanIntensity(pixelBuffer: pixelBuffer)
        
        // 350 is the number of frames we choose to analyze
        if (pixelIntensities.count == 350) {
            var candidate_bits : [[Double]] = []
            for i in (0...350-windowSize) {
                let input = Array<Double>(pixelIntensities![i...i + windowSize - 1])
                let patterns = checkPattern(input: input, windowSize: windowSize)
                candidate_bits.append(patterns)
                print("Score zero: ", patterns[0])
                print("Score ones: ", patterns[1])
                print("-------")
            }
            let start_idx = findStart(inputs: candidate_bits)
            print("Final Sequence: ")
            var result: String = ""
            for i in (0...seqLength - 1){
                let curr_idx = start_idx + i * windowSize
                if ((candidate_bits[curr_idx][0] > candidate_bits[curr_idx][1])) {
                    print("0")
                    result.append("0")
                    print((candidate_bits[curr_idx][0]))
                    print("---------")
                }
                else if (candidate_bits[curr_idx][0] < candidate_bits[curr_idx][1] ) {
                    result.append("1")
                    print("1")
                    print((candidate_bits[curr_idx][1]))
                    print("---------")
                }
            }
            transmittedText = binaryToAscii(input: result)
        } else{
             pixelIntensities.append(x)
        }
        print("Camera was able to caputre a frame:", Date())
    }

    func findStart(inputs: [[Double]]) -> Int {
        // Looks for the best start frame for the encoded sequence
        var best_idx = -1
        var best_score = -10000.0
        for start_idx in (0...inputs.count - (seqLength * windowSize)) {
            var curr_score = 0.0
            for i in (0...seqLength - 1) {
                let curr_idx = start_idx + i * windowSize
                curr_score += max(inputs[curr_idx][0], inputs[curr_idx][1])
            }
            if (curr_score > best_score) {
                best_score = curr_score
                best_idx = start_idx
            }
        }
        return best_idx
    }
}

func meanIntensity(pixelBuffer: CVPixelBuffer) -> Double {
    // Simple average intensity of a pixel buffer
    var sum = 0.0
    var count = 0
    let skip = 200
    let data_size = CVPixelBufferGetDataSize(pixelBuffer)
    
    CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
    let baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
    CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)

    for i in stride(from: 0, to: data_size, by: skip) {
        let offsetPointer = baseAddress! + i
        sum += Double(offsetPointer.load(as: UInt8.self))
        count += 1
    }
    
    return sum / Double(count)
}

func checkPattern(input: [Double], windowSize: Int) -> [Double] {
    // Pattern matching to see whether a window is a 0 or 1
    var ones: [Double]
    var zeros: [Double]
    if (windowSize == 12) {
        ones = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
        zeros = [-0.66666, -0.66666, 1.33333, -0.66666, -0.66666, 1.33333, -0.66666, -0.66666, 1.33333, -0.66666, -0.66666, 1.33333]
    }
    else {
        ones = [-1, 1, -1, 1, -1, 1]
        zeros = [-0.66666, -0.66666, 1.33333, -0.66666, -0.66666, 1.33333]
    }

    var score_ones = 0.0
    var score_zeros = 0.0
    
    for i in (0...windowSize - 1) {
        score_ones += input[i] * ones[i]
        score_zeros += input[i] * zeros[i]
    }
    
    return [score_zeros, score_ones]
}

// Code from https://stackoverflow.com/questions/42668013/swift-convert-a-binary-string-to-its-ascii-values
func binaryToAscii(input: String) -> String {
    var index = input.startIndex
    var result: String = ""
    for _ in 0...(input.count/8 - 1) {
        let nextIndex = input.index(index, offsetBy: 8)
        let charBits = input[index..<nextIndex]
        result += String(UnicodeScalar(UInt8(charBits, radix: 2)!))
        index = nextIndex
    }
    print(result)
    return(result) //->Hey
}

func configureCameraForHighestFrameRate(device: AVCaptureDevice) {
    // Some parameters like choosing the best frame rate and fixing the
    // focus and exposure
    var bestFormat: AVCaptureDevice.Format?
    var bestFrameRateRange: AVFrameRateRange?

    for format in device.formats {
        for range in format.videoSupportedFrameRateRanges {
            if range.maxFrameRate > bestFrameRateRange?.maxFrameRate ?? 0{
                print(range.maxFrameRate)
                bestFormat = format
                bestFrameRateRange = range
            }
        }
    }

    if let bestFormat = bestFormat,
       let bestFrameRateRange = bestFrameRateRange {
        do {
            try device.lockForConfiguration()

            // Set the device's active format.
            device.activeFormat = bestFormat

            device.exposureMode = AVCaptureDevice.ExposureMode.locked
            device.focusMode = AVCaptureDevice.FocusMode.locked

            // Set the device's min/max frame duration.
            let duration = bestFrameRateRange.minFrameDuration
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration

            device.unlockForConfiguration()
        } catch {
            // Handle error.
        }
    }
}


