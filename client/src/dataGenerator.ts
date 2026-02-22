// client/src/dataGenerator.ts
import * as tf from '@tensorflow/tfjs';

export const generatePersonaData = (persona: 'athlete' | 'senior' | 'average') => {
    const numSamples = 100;
    const data = [];
    const labels = [];

    let noiseMultiplier = persona === 'athlete' ? 0.5 : persona === 'senior' ? 0.1 : 0.2;

    for (let i = 0; i < numSamples; i++) {
        const isFall = Math.random() > 0.9; // 10% chance of an event
        const isHardSit = Math.random() > 0.8 && !isFall; // 20% chance of false alarm trigger

        if (isFall) {
            // Fall: Freefall followed by massive spike
            data.push([Math.random() * 0.1, Math.random() * 0.1, 9.8 + Math.random() * 5]);
            labels.push([0, 0, 1]); // Class 2: Fall
        } else if (isHardSit) {
            // Hard Sit: Sudden medium spike
            data.push([Math.random() * noiseMultiplier, 2.0 + Math.random(), Math.random()]);
            labels.push([0, 1, 0]); // Class 1: Hard Sit (False Alarm trap)
        } else {
            // Normal Walking: Sine wave pattern
            data.push([
                Math.sin(i) * noiseMultiplier, 
                Math.cos(i) * noiseMultiplier, 
                1.0 + Math.random() * noiseMultiplier
            ]);
            labels.push([1, 0, 0]); // Class 0: Normal ADL
        }
    }

    return {
        xs: tf.tensor2d(data, [numSamples, 3]),
        ys: tf.tensor2d(labels, [numSamples, 3])
    };
};