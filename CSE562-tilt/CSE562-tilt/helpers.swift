//
//  helpers.swift
//  CSE562-tilt
//
//  Created by Vivek Jayaram on 5/6/20.
//  Copyright © 2020 Vivek Jayaram. All rights reserved.
//

import UIKit
import simd

class Helpers {
    
    func createQuaternion(v: [Double], theta: Double) -> simd_quatd {
        // Creates a quaternion wiht rotation theta around a normalized vector v
        
        let rotationQuaternion = simd_quatd(ix: v[0] * sin(theta / 2.0),
                                            iy: v[1] * sin(theta / 2.0),
                                            iz: v[2] * sin(theta / 2.0),
                                            r: cos(theta / 2.0))
        return rotationQuaternion
    }
    
    func updateGyroOnly(currQ: simd_quatd, w: [Double], deltaT: Double) -> simd_quatd {
        // Updates our quaternion using only gyroscope data
        let length = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).squareRoot()
        let v = [-w[0] / length, w[1] / length, -w[2] / length]
        
        let rotationQuaternion = createQuaternion(v: v, theta: length * deltaT)

        let result = simd_mul(currQ, rotationQuaternion)
        
        return result
    }
    
    func updateAccelOnly(currQ: simd_quatd, w: [Double], alpha: Double) -> simd_quatd {
        // Updates our quaternion using only acceleromoter data
        let length = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).squareRoot()
        let a = [w[0] / length, w[1] / length, w[2] / length]
        
        
        let accelQuaternion = simd_quatd(ix: a[0], iy: a[1], iz: a[2], r: 0)
        let global_a = simd_mul(currQ.inverse, simd_mul(accelQuaternion, currQ))
        
        let accelAxisProj = [global_a.axis[0], 0, global_a.axis[2]]

        var tiltAxis = [accelAxisProj[2], 0, -1.0 * accelAxisProj[0]]

        let tiltAngle = angleBetweenVecs(a: [global_a.axis[0], global_a.axis[1], global_a.axis[2]], b: [0.0, -1.0, 0.0])
        
        let tiltAxisLen = (tiltAxis[0] * tiltAxis[0] + tiltAxis[2] * tiltAxis[2]).squareRoot()
        tiltAxis = [tiltAxis[0] / tiltAxisLen, 0, tiltAxis[2] / tiltAxisLen]
        
        var result = createQuaternion(v: tiltAxis, theta: -alpha * tiltAngle)

        return result
    }

    func updateBoth(currQ: simd_quatd, gyro: [Double], accel: [Double], deltaT: Double) -> simd_quatd {
        let gyroResult = updateGyroOnly(currQ: currQ, w: gyro, deltaT: deltaT)
        let accelResult = updateAccelOnly(currQ: currQ, w: accel, alpha: 0.01)
        
        var result = simd_mul(accelResult, gyroResult)

        return result
    }
    
    func angleBetweenVecs(a: [Double], b: [Double]) -> Double {
        // Returns the angle between two 3d vectors
        let dotProd = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        let magA = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).squareRoot()
        let magB = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).squareRoot()
        
        return acos(dotProd / (magA * magB))
    }
    
    
//    func quatToRoll(q: simd_quatd) -> Double {
//        // Takes a quaternion and returns the roll parameter
//        var roll = atan2(2.0*(q.imag[0] * q.imag[1] + q.real * q.imag[2]), q.real * q.real + q.imag[0] * q.imag[0] - q.imag[1] * q.imag[1] - q.imag[2] * q.imag[2]);
//
//        return roll
//    }
//
//    func quatToPitch(q: simd_quatd) -> Double {
//        // Takes a quaternion and returns the roll parameter
//        var pitch = asin(-2.0*(q.imag[0] * q.imag[2] - q.real * q.imag[1]))
//        return pitch
//    }
    
    func quatToPitch(q: simd_quatd) -> Double {
        // Takes a quaternion and returns the roll parameter
        var pitch = atan2(2.0*(q.imag[2] * q.imag[1] + q.real * q.imag[0]), q.real * q.real - q.imag[0] * q.imag[0] - q.imag[2] * q.imag[2] + q.imag[1] * q.imag[1]);
        return pitch
    }

    func quatToRoll(q: simd_quatd) -> Double {
        // Takes a quaternion and returns the roll parameter
        var pitch = asin(-2.0*(q.imag[0] * q.imag[1] - q.real * q.imag[2]))
        return pitch
    }

}
