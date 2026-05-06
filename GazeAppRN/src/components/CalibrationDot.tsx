import React, { useEffect, useRef } from 'react';
import { View, Text, Animated, StyleSheet } from 'react-native';
import Svg, { Circle } from 'react-native-svg';

interface Props {
  fixationLetter: string;
  progress: number;      // 0..1, fraction of point duration elapsed
  isCollecting: boolean;
  dotRadius: number;
  color: string;         // hex — green for calibration, yellow for validation
}

export default function CalibrationDot({
  fixationLetter,
  progress,
  isCollecting,
  dotRadius,
  color,
}: Props) {
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const loopRef = useRef<Animated.CompositeAnimation | null>(null);

  const ringSize = (dotRadius + 12) * 2;
  const radius = dotRadius + 8;           // SVG circle radius (inside stroke)
  const circumference = 2 * Math.PI * radius;
  const lineWidth = 4;
  const center = ringSize / 2;

  useEffect(() => {
    if (isCollecting) {
      loopRef.current = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 500,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1.0,
            duration: 500,
            useNativeDriver: true,
          }),
        ]),
      );
      loopRef.current.start();
    } else {
      loopRef.current?.stop();
      Animated.timing(pulseAnim, {
        toValue: 1.0,
        duration: 80,
        useNativeDriver: true,
      }).start();
    }
  }, [isCollecting]);

  const strokeDashoffset = circumference * (1 - (isCollecting ? progress : 1.0));

  return (
    <View style={[styles.wrapper, { width: ringSize, height: ringSize }]}>
      {/* SVG progress ring */}
      <Svg width={ringSize} height={ringSize} style={StyleSheet.absoluteFill}>
        {/* Background ring */}
        <Circle
          cx={center}
          cy={center}
          r={radius}
          stroke="rgba(255,255,255,0.2)"
          strokeWidth={lineWidth}
          fill="none"
        />
        {/* Animated progress arc */}
        <Circle
          cx={center}
          cy={center}
          r={radius}
          stroke={color}
          strokeWidth={lineWidth}
          fill="none"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          rotation="-90"
          origin={`${center},${center}`}
          opacity={isCollecting ? 1 : 0.55}
        />
      </Svg>

      {/* Pulsing center dot */}
      <Animated.View
        style={[
          styles.centerDot,
          {
            width: dotRadius * 2,
            height: dotRadius * 2,
            borderRadius: dotRadius,
            backgroundColor: isCollecting ? '#ffffff' : color + 'b3',
            transform: [{ scale: pulseAnim }],
          },
        ]}
      />

      {/* Fixation letter */}
      <Text style={styles.letter}>{fixationLetter}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  centerDot: {
    position: 'absolute',
  },
  letter: {
    position: 'absolute',
    fontSize: 15,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    color: '#000',
  },
});
