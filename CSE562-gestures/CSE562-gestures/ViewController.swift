//
//  ViewController.swift
//  CSE562-gestures
//
//  Created by Vivek Jayaram on 4/13/20.
//  Copyright © 2020 Vivek Jayaram. All rights reserved.
//

import UIKit
import AudioKit
import Charts
import Foundation

class ViewController: UIViewController {
    @IBOutlet weak var gesture: UILabel!
    @IBOutlet weak var lineChartView: LineChartView!

    // Pilot tone constants
    var oscillator = AKOscillator()
    var mainMixer = AKMixer()
    let TARGET_FREQ = 18000.0  // Our pilot tone
    let SAMPLE_RATE = 44100.0  // Input and output sample rate
    let SOUND_SPEED = 340.0  // m/s

    // Mic constants
    var mic: AKMicrophone!
    var micMixer: AKMixer!
    var micBooster: AKBooster!
    var fftTap: AKFFTTap!
    let FFT_SIZE = 512

    // Calibration constants
    var calibrationSteps = 0  // Keep track of the number of calibration steps
    var calibrationAmplitudes: [[Double]]!  // Calculate the average dB before gestures
    var baselineAmplitudes: [Double]!
    var calibrationFinished = false
    let CALIBRATION_TIME = 2.0  // Number of seconds to calibrate the non-gesture dB levels

    // Gesture constants
    let TIME_INTERVAL = 0.01  // How often to do the gesture check
    let MIN_HAND_SPEED = 1.0  // meters per sec
    let MAX_HAND_SPEED = 15.0  // meters per sec
    let MIN_DB_GAIN = 20.0  // Target frequency bins must increase this much to have a gesture triggered
 

    override func viewDidLoad() {
        super.viewDidLoad()
        do {
            try AKSettings.setSession(category: .playAndRecord, with: .allowBluetoothA2DP)
        } catch {
            AKLog("Could not set session category.")
        }
        AKSettings.defaultToSpeaker = true

        AKSettings.sampleRate = SAMPLE_RATE
        AKSettings.audioInputEnabled = true
        mic = AKMicrophone()
        micMixer = AKMixer(mic)
        micBooster = AKBooster(micMixer)
        micBooster.gain = 0
        fftTap = AKFFTTap.init(mic)
        initializeBaselines()
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        mainMixer = AKMixer(oscillator, micBooster)
        AudioKit.output = mainMixer

        do {
            try AudioKit.start()
        } catch {
            AKLog("AudioKit did not start!")
        }

        oscillator.frequency = TARGET_FREQ
        oscillator.start()


        mic.start()
        Timer.scheduledTimer(timeInterval: TIME_INTERVAL,
                             target: self,
                             selector: #selector(ViewController.updateUI),
                             userInfo: nil,
                             repeats: true)
    }
    
    @objc func updateUI() {
        var frequencies: [Double] = []
        var amplitudes: [Double] = []
        for i in stride(from: 0, to: FFT_SIZE, by: 2) {
            let re = self.fftTap!.fftData[i]
            let im = self.fftTap!.fftData[i + 1]
            let normBinMag = 2.0 * sqrt(re * re + im * im)/self.FFT_SIZE

            let amplitude = ((20.0 * log10(normBinMag)))

            let freq = binToFreq(bin: i)
            frequencies.append(freq)
            amplitudes.append(amplitude)
            // print("bin: \(i/2) \t freq:\(freq) \t ampl.: \(amplitude)")
        }
        updateChart(frequencies: frequencies, amplitudes: amplitudes)
        
        // Do some number of calibration steps to see what is a good baseline
        if (Double(calibrationSteps) < CALIBRATION_TIME / TIME_INTERVAL) {
            calibrationStep(amplitudes: amplitudes)
            calibrationSteps += 1
        } else{
            if (!calibrationFinished) {
                computeBaselines()
            }

            // The actual gesture recognition algorithm
            detectGestures(frequencies: frequencies, amplitudes: amplitudes)
        }
        
    }
    
    func updateChart(frequencies: [Double], amplitudes: [Double]) {
        // Draw the FFT chart with the values
        var values: [ChartDataEntry] = []
        for i in (0...Int(Double(FFT_SIZE)/2) - 1){
            values.append(ChartDataEntry(x: frequencies[i], y: amplitudes[i]))
        }
        
        let set1 = LineChartDataSet(entries: values, label: "FFT")
        let data = LineChartData(dataSet: set1)
        self.lineChartView.data = data
        self.lineChartView.leftAxis.axisMinimum = -240
        self.lineChartView.rightAxis.axisMinimum = -240

        self.lineChartView.leftAxis.axisMaximum = -60
        self.lineChartView.rightAxis.axisMaximum = -60
    }
    
    func calibrationStep(amplitudes: [Double]) {
        // Add the observed amplitude for every frequency bin to an array
        for i in (0...Int(Double(FFT_SIZE)/2) - 1) {
            calibrationAmplitudes[i].append(amplitudes[i])
        }
    }

    func initializeBaselines() {
        // Initialize the average amplitudes for calibration
        calibrationAmplitudes = []
        for _ in (0...Int(Double(FFT_SIZE)/2) - 1) {
            calibrationAmplitudes.append([])
        }
    }

    func binToFreq(bin: Int) -> Double {
        return SAMPLE_RATE * 0.5 * Double(bin) / FFT_SIZE
    }
    
    func freqToBin(frequency: Double) -> Int {
        return Int(round(frequency * FFT_SIZE / (SAMPLE_RATE * 0.5)))
    }

    func computeBaselines(){
        // Calculate the median amplitude across bins
        calibrationFinished = true
        baselineAmplitudes = []
        for i in (0...Int(Double(FFT_SIZE)/2) - 1) {
            baselineAmplitudes.append(calculateMedian(array: calibrationAmplitudes[i]))
        }
    }
    
    func calculateMedian(array: [Double]) -> Double {
        let sorted = array.sorted()
        if sorted.count % 2 == 0 {
            return Double((sorted[(sorted.count / 2)] + sorted[(sorted.count / 2) - 1])) / 2
        } else {
            return Double(sorted[(sorted.count - 1) / 2])
        }
    }

    func detectGestures(frequencies: [Double], amplitudes: [Double]) {
        // Calculate the frequency bins of interest using the doppler shift formula
        let minPullBin = freqToBin(frequency: TARGET_FREQ * (SOUND_SPEED - MAX_HAND_SPEED) / (SOUND_SPEED + MAX_HAND_SPEED)) / 2
        let maxPullBin = freqToBin(frequency: TARGET_FREQ * (SOUND_SPEED - MIN_HAND_SPEED) / (SOUND_SPEED + MIN_HAND_SPEED)) / 2
        
        let minPushBin = freqToBin(frequency: TARGET_FREQ * (SOUND_SPEED + MIN_HAND_SPEED) / (SOUND_SPEED - MIN_HAND_SPEED)) / 2
        let maxPushBin = freqToBin(frequency: TARGET_FREQ * (SOUND_SPEED + MAX_HAND_SPEED) / (SOUND_SPEED - MAX_HAND_SPEED)) / 2
        
        // Check for a pull gesture. We assume a simple thresholding function.
        for bin_idx in (minPullBin...maxPullBin) {
            if (amplitudes[bin_idx] > MIN_DB_GAIN + baselineAmplitudes[bin_idx]) {
                gesture.text = "Pull"
            }
        }
        
        // Check for a push gesture
        for bin_idx in (minPushBin...maxPushBin) {
            if (amplitudes[bin_idx] > MIN_DB_GAIN + baselineAmplitudes[bin_idx]) {
                gesture.text = "Push"
            }
        }
    }
}

