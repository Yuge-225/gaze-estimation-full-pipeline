import React, { useRef, useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { Camera, useCameraDevice, useCameraFormat } from 'react-native-vision-camera';
import RNFS from 'react-native-fs';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';
import { CalibrationConfig, CameraConfig } from '../constants/config';
import { EventLogger } from '../services/EventLogger';
import CalibrationDot from '../components/CalibrationDot';
import { createNewSession } from '../models/CalibrationSession';
import type { CalibrationSession } from '../models/CalibrationSession';

type Props = NativeStackScreenProps<RootStackParamList, 'Calibration'>;

const { width: W, height: H, scale: SCALE } = Dimensions.get('window');

function randomLetter(): string {
  return CalibrationConfig.fixationLetters[
    Math.floor(Math.random() * CalibrationConfig.fixationLetters.length)
  ];
}

export default function CalibrationScreen({ navigation, route }: Props) {
  const { videoUri, videoFilename } = route.params;
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [{ fps: CameraConfig.frameRate }]);
  const cameraRef = useRef<Camera>(null);
  const loggerRef = useRef<EventLogger | null>(null);
  const sessionRef = useRef<CalibrationSession | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isRecording = useRef(false);
  const pointIndexRef = useRef(0);
  const startTimeRef = useRef(0);

  const [pointIndex, setPointIndex] = useState(0);
  const [fixationLetter, setFixationLetter] = useState('A');
  const [countdown, setCountdown] = useState(CalibrationConfig.pointDuration);
  const [isCollecting, setIsCollecting] = useState(false);

  const finishCalibration = useCallback(async () => {
    const logger = loggerRef.current;
    const session = sessionRef.current;
    if (!logger || !session) return;

    const sw = Math.round(W * SCALE);
    const sh = Math.round(H * SCALE);
    const dr = Math.round(CalibrationConfig.dotRadius * SCALE);
    logger.log(`${logger.elapsedMs()},session_end,-1,-,-,-,-,${sw},${sh},${dr},-`);

    if (isRecording.current) {
      isRecording.current = false;
      await cameraRef.current?.stopRecording();
    }
    await logger.save();

    navigation.navigate('Validation', { session, videoUri, videoFilename });
  }, [navigation, videoUri, videoFilename]);

  const beginPoint = useCallback(
    (index: number) => {
      if (index >= CalibrationConfig.pointCount) {
        finishCalibration();
        return;
      }

      const pos = CalibrationConfig.points[index];
      const letter = randomLetter();
      pointIndexRef.current = index;

      setPointIndex(index);
      setFixationLetter(letter);
      setCountdown(CalibrationConfig.pointDuration);
      setIsCollecting(true);

      const sw = Math.round(W * SCALE);
      const sh = Math.round(H * SCALE);
      const dr = Math.round(CalibrationConfig.dotRadius * SCALE);
      const txpx = Math.round(pos.x * sw);
      const typx = Math.round(pos.y * sh);
      const ms = loggerRef.current?.elapsedMs() ?? 0;
      loggerRef.current?.log(
        `${ms},point_start,${index},` +
          `${pos.x.toFixed(4)},${pos.y.toFixed(4)},` +
          `${txpx},${typx},${sw},${sh},${dr},${letter}`,
      );

      startTimeRef.current = Date.now();
      let tick = 0;

      timerRef.current = setInterval(() => {
        tick++;
        // Rotate fixation letter every 10 ticks (0.5 s)
        if (tick % 10 === 0) {
          const newLetter = randomLetter();
          setFixationLetter(newLetter);
        }

        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        const remaining = Math.max(0, CalibrationConfig.pointDuration - elapsed);
        setCountdown(remaining);

        if (elapsed >= CalibrationConfig.pointDuration) {
          clearInterval(timerRef.current!);
          timerRef.current = null;
          setIsCollecting(false);
          setTimeout(
            () => beginPoint(pointIndexRef.current + 1),
            CalibrationConfig.transitionDelay * 1000,
          );
        }
      }, 50);
    },
    [finishCalibration],
  );

  useEffect(() => {
    let mounted = true;

    const setup = async () => {
      const session = await createNewSession();
      sessionRef.current = session;

      const logger = new EventLogger(
        session.calibrationEventsPath,
        'elapsed_ms,event_type,point_index,target_x_norm,target_y_norm,' +
          'target_x_px,target_y_px,screen_w_px,screen_h_px,dot_radius_px,fixation_letter',
      );
      loggerRef.current = logger;

      // Give the camera ~1 s to auto-expose/focus before locking and recording
      setTimeout(async () => {
        if (!mounted) return;
        isRecording.current = true;

        await cameraRef.current?.startRecording({
          fileType: 'mp4',
          onRecordingFinished: async video => {
            try {
              await RNFS.copyFile(video.path, session.calibrationVideoPath);
            } catch {}
          },
          onRecordingError: e => console.error('[Calibration] recording error:', e),
        });

        logger.markStart();
        const sw = Math.round(W * SCALE);
        const sh = Math.round(H * SCALE);
        const dr = Math.round(CalibrationConfig.dotRadius * SCALE);
        logger.log(`0,session_start,-1,-,-,-,-,${sw},${sh},${dr},-`);
        beginPoint(0);
      }, 1000);
    };

    setup();

    return () => {
      mounted = false;
      if (timerRef.current) clearInterval(timerRef.current);
      if (isRecording.current) {
        isRecording.current = false;
        cameraRef.current?.stopRecording();
      }
      loggerRef.current?.save();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const clampedIndex = Math.min(pointIndex, CalibrationConfig.pointCount - 1);
  const pos = CalibrationConfig.points[clampedIndex];
  const progress = 1.0 - countdown / CalibrationConfig.pointDuration;

  return (
    <View style={styles.container}>
      {/* Hidden camera — records the front-facing video; no preview shown */}
      {device && format ? (
        <Camera
          ref={cameraRef}
          style={styles.hiddenCamera}
          device={device}
          isActive={true}
          video={true}
          format={format}
          fps={CameraConfig.frameRate}
          videoStabilizationMode="off"
        />
      ) : null}

      <Text style={styles.progress}>
        Calibration {'  '}
        {pointIndex + 1} / {CalibrationConfig.pointCount}
      </Text>

      {/* Calibration dot — positioned at target coordinates */}
      <View
        style={[
          styles.dotWrapper,
          {
            left: pos.x * W - (CalibrationConfig.dotRadius + 12),
            top: pos.y * H - (CalibrationConfig.dotRadius + 12),
          },
        ]}
      >
        <CalibrationDot
          fixationLetter={fixationLetter}
          progress={progress}
          isCollecting={isCollecting}
          dotRadius={CalibrationConfig.dotRadius}
          color="#4caf50"
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  hiddenCamera: { width: 1, height: 1, position: 'absolute', opacity: 0 },
  progress: {
    position: 'absolute',
    top: 24,
    alignSelf: 'center',
    color: 'rgba(255,255,255,0.7)',
    fontSize: 16,
    fontWeight: '600',
  },
  dotWrapper: { position: 'absolute' },
});
