//
//  ViewController.swift
//  CSE562-tilt
//
//  Created by Vivek Jayaram on 5/1/20.
//  Copyright © 2020 Vivek Jayaram. All rights reserved.
//

import UIKit
import CoreMotion
import simd

class ViewController: UIViewController {
    
    var motionManager: CMMotionManager!
    var timer: Timer!
    
    var q: simd_quatd! // Quaternion representing our current position
    var tilt = 0.0  // Keep track of the tilt for printing purposes

    // This is for measuring the noise and variance of the sensors
    var baselineAccel: [[Double]]!
    var baselineGyro: [[Double]]!
    let CALIBRATION_TIME = 5.0
    var calibrationSteps = 0  // Keep track of the number of steps
    let UPDATE_INTERVAL = 0.01  // Update 1000 times a second
    
    // For bias correction, using our measured biases
    let accelBias = [0.00757, -0.00343, 0.00694]
    let gyroBias = [-0.00207, 0.00245, -0.00534]
    
    // Submodule
    var helpers = Helpers()
    
    @IBOutlet weak var accel_x: UILabel!
    @IBOutlet weak var accel_y: UILabel!
    @IBOutlet weak var accel_z: UILabel!
    @IBOutlet weak var gyro_x: UILabel!
    @IBOutlet weak var gyro_y: UILabel!
    @IBOutlet weak var gyro_z: UILabel!

    @IBOutlet weak var roll_label: UILabel!
    @IBOutlet weak var pitch_label: UILabel!
    @IBOutlet weak var axis_label: UILabel!
    @IBOutlet weak var angle_label: UILabel!

    @IBOutlet weak var demo_pic: UIImageView!
    
    var prevRoll = 0.0  // For the picture demo
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        demo_pic.image = UIImage(named: "morges_sunset_circle.png")!

        // Initialize our quaternion as the identity
        q = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
        
        // Initialize our calibration measurements
        baselineAccel = []
        baselineGyro = []
        for _ in (0...3){
            baselineAccel.append([])
            baselineGyro.append([])
        }
        
        // Do additional setup after loading the view.
        motionManager = CMMotionManager()
        motionManager.startAccelerometerUpdates()
        motionManager.startGyroUpdates()

        Timer.scheduledTimer(timeInterval: UPDATE_INTERVAL,
                             target: self,
                             selector: #selector(ViewController.updateStep),
                             userInfo: nil,
                             repeats: true)
    }

    @objc func updateStep() {
        
        // Only for calibration
        if (Double(calibrationSteps) * UPDATE_INTERVAL == CALIBRATION_TIME){
            // If you want to measure bias and variance then run the following line
            //runBiasVariance()
        }

        // Knowing when to stop calibration
        calibrationSteps += 1

        if (self.motionManager.isAccelerometerAvailable && self.motionManager.isGyroAvailable) {
            if let accelData = motionManager.accelerometerData {
                if let gyroData = motionManager.gyroData {
                    
                    // Get accelerometer data
                    let accelX = accelData.acceleration.x - accelBias[0]
                    let accelY = accelData.acceleration.y - accelBias[1]
                    let accelZ = accelData.acceleration.z - accelBias [2]
                    
                    accel_x.text = String(format: "Accelerometer x: %.5f", accelX)
                    accel_y.text = String(format: "Accelerometer y: %.5f", accelY)
                    accel_z.text = String(format: "Accelerometer z: %.5f", accelZ)
                    
                    baselineAccel[0].append(accelX)
                    baselineAccel[1].append(accelY)
                    baselineAccel[2].append(accelZ)
                        
                    // Get gyroscope data
                    let gyroX = gyroData.rotationRate.x - gyroBias[0]
                    let gyroY = gyroData.rotationRate.y - gyroBias[1]
                    let gyroZ = gyroData.rotationRate.z - gyroBias[2]

                    gyro_x.text = String(format: "Gyro x: %.5f", gyroX)
                    gyro_y.text = String(format: "Gyro y: %.5f", gyroY)
                    gyro_z.text = String(format: "Gyro z: %.5f", gyroZ)
                    
                    baselineGyro[0].append(gyroX)
                    baselineGyro[1].append(gyroY)
                    baselineGyro[2].append(gyroZ)
                    
                    // Update with only the gyroscope
                    // q = helpers.updateGyroOnly(currQ: q, w: [gyroX, gyroY, gyroZ], deltaT: UPDATE_INTERVAL)
                    
                    // Update with only accelerometer
                    // q = simd_mul(helpers.updateAccelOnly(currQ: q, w: [accelX, accelY, accelZ], alpha:1.0), q)

                    // Update with both gyroscope and accelerometer
                    q = helpers.updateBoth(currQ: q,
                                           gyro: [gyroX, gyroY, gyroZ],
                                           accel: [accelX, accelY, accelZ],
                                           deltaT: UPDATE_INTERVAL)

                    // Calculate things we need
                    let curr_roll = helpers.quatToRoll(q: q)
                    demo_pic.transform = demo_pic.transform.rotated(by: CGFloat(prevRoll - curr_roll))
                    prevRoll = curr_roll
                    roll_label.text = String(format: "Roll: %.1f", curr_roll  * 180 / 3.14159)
                    
                    let curr_pitch = helpers.quatToPitch(q: q)
                    pitch_label.text = String(format: "Pitch: %.1f", curr_pitch  * 180 / 3.14159)
                
                    axis_label.text = String(format: "Axis: [%.1f, %.1f]", q.axis[0], q.axis[2])
                    
                    tilt = acos(cos(curr_roll) * cos(curr_pitch))
                    angle_label.text = String(format: "Tilt Angle: %.1f Degrees", tilt * 180 / 3.1415926535)
                }
            }
        }
        
        // Printing the tilt for graphing purposes
        if (calibrationSteps % 50 == 0) {
            print(String(format: "%.4f,", tilt))
        }
    }
    
    func calculateMean(inputArray: [Double]) -> Double {
        // Simple mean function
        let length = inputArray.count
        var sum = 0.0
        for i in (0...length - 1) {
            sum += inputArray[i]
        }
        
        return sum / Double(length)
    }
    
    func calculateVariance(inputArray: [Double], mean: Double) -> Double {
        // Simple average squared error
        let length = inputArray.count
        var sum = 0.0
        for i in (0...length - 1) {
            let diff = (inputArray[i] - mean)
            sum += diff * diff
        }
        
        return sum / Double(length)
    }
    
    func runBiasVariance() {
        // Calculates the bias and variance for all the different sensors

        var x_mean = calculateMean(inputArray: baselineAccel[0])
        print(String(format: "Accelerometer x mean: %.10f", x_mean))
        print(String(format: "Accelerometer x variance: %.10f", calculateVariance(inputArray: baselineAccel[0], mean: x_mean)))

        var y_mean = calculateMean(inputArray: baselineAccel[1])
        print(String(format: "Accelerometer y mean: %.10f", y_mean))
        print(String(format: "Accelerometer y variance: %.10f", calculateVariance(inputArray: baselineAccel[1], mean: y_mean)))
        
        var z_mean = calculateMean(inputArray: baselineAccel[2])
        print(String(format: "Accelerometer z mean: %.10f", z_mean))
        print(String(format: "Accelerometer z variance: %.10f", calculateVariance(inputArray: baselineAccel[2], mean: z_mean)))
        
        x_mean = calculateMean(inputArray: baselineGyro[0])
        print(String(format: "Gyro x mean: %.10f", x_mean))
        print(String(format: "Gyro x variance: %.10f", calculateVariance(inputArray: baselineGyro[0], mean: x_mean)))

        y_mean = calculateMean(inputArray: baselineGyro[1])
        print(String(format: "Gyro y mean: %.10f", y_mean))
        print(String(format: "Gyro y variance: %.10f", calculateVariance(inputArray: baselineGyro[1], mean: y_mean)))

        z_mean = calculateMean(inputArray: baselineGyro[2])
        print(String(format: "Gyro z mean: %.10f", z_mean))
        print(String(format: "Gyro z variance: %.10f", calculateVariance(inputArray: baselineGyro[2], mean: z_mean)))
    }

}

