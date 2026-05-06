import React, { useRef, useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Dimensions } from 'react-native';
import { Camera, useCameraDevice, useCameraFormat } from 'react-native-vision-camera';
import RNFS from 'react-native-fs';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';
import { ValidationConfig, CalibrationConfig, CameraConfig } from '../constants/config';
import { EventLogger } from '../services/EventLogger';
import CalibrationDot from '../components/CalibrationDot';

type Props = NativeStackScreenProps<RootStackParamList, 'Validation'>;

const { width: W, height: H, scale: SCALE } = Dimensions.get('window');

function randomLetter(): string {
  return CalibrationConfig.fixationLetters[
    Math.floor(Math.random() * CalibrationConfig.fixationLetters.length)
  ];
}

export default function ValidationScreen({ navigation, route }: Props) {
  const { session, videoUri, videoFilename } = route.params;
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [{ fps: CameraConfig.frameRate }]);
  const cameraRef = useRef<Camera>(null);
  const loggerRef = useRef<EventLogger | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isRecording = useRef(false);
  const pointIndexRef = useRef(0);
  const startTimeRef = useRef(0);

  const [phase, setPhase] = useState<'collecting' | 'done'>('collecting');
  const [pointIndex, setPointIndex] = useState(0);
  const [fixationLetter, setFixationLetter] = useState('A');
  const [countdown, setCountdown] = useState(ValidationConfig.pointDuration);
  const [isCollecting, setIsCollecting] = useState(false);

  const finishValidation = useCallback(async () => {
    const logger = loggerRef.current;
    if (!logger) return;

    const sw = Math.round(W * SCALE);
    const sh = Math.round(H * SCALE);
    const dr = Math.round(ValidationConfig.dotRadius * SCALE);
    logger.log(`${logger.elapsedMs()},session_end,-1,-,-,-,-,${sw},${sh},${dr},-`);

    if (isRecording.current) {
      isRecording.current = false;
      await cameraRef.current?.stopRecording();
    }
    await logger.save();
    setPhase('done');
  }, []);

  const beginPoint = useCallback(
    (index: number) => {
      if (index >= ValidationConfig.pointCount) {
        finishValidation();
        return;
      }

      const pos = ValidationConfig.points[index];
      const letter = randomLetter();
      pointIndexRef.current = index;

      setPointIndex(index);
      setFixationLetter(letter);
      setCountdown(ValidationConfig.pointDuration);
      setIsCollecting(true);

      const sw = Math.round(W * SCALE);
      const sh = Math.round(H * SCALE);
      const dr = Math.round(ValidationConfig.dotRadius * SCALE);
      const ms = loggerRef.current?.elapsedMs() ?? 0;
      loggerRef.current?.log(
        `${ms},point_start,${index},` +
          `${pos.x.toFixed(4)},${pos.y.toFixed(4)},` +
          `${Math.round(pos.x * sw)},${Math.round(pos.y * sh)},${sw},${sh},${dr},${letter}`,
      );

      startTimeRef.current = Date.now();
      let tick = 0;

      timerRef.current = setInterval(() => {
        tick++;
        if (tick % 10 === 0) setFixationLetter(randomLetter());

        const elapsed = (Date.now() - startTimeRef.current) / 1000;
        setCountdown(Math.max(0, ValidationConfig.pointDuration - elapsed));

        if (elapsed >= ValidationConfig.pointDuration) {
          clearInterval(timerRef.current!);
          timerRef.current = null;
          setIsCollecting(false);
          setTimeout(
            () => beginPoint(pointIndexRef.current + 1),
            ValidationConfig.transitionDelay * 1000,
          );
        }
      }, 50);
    },
    [finishValidation],
  );

  useEffect(() => {
    let mounted = true;

    const logger = new EventLogger(
      session.validationEventsPath,
      'elapsed_ms,event_type,point_index,target_x_norm,target_y_norm,' +
        'target_x_px,target_y_px,screen_w_px,screen_h_px,dot_radius_px,fixation_letter',
    );
    loggerRef.current = logger;

    setTimeout(async () => {
      if (!mounted) return;
      isRecording.current = true;

      await cameraRef.current?.startRecording({
        fileType: 'mp4',
        onRecordingFinished: async video => {
          try {
            await RNFS.copyFile(video.path, session.validationVideoPath);
          } catch {}
        },
        onRecordingError: e => console.error('[Validation] recording error:', e),
      });

      logger.markStart();
      const sw = Math.round(W * SCALE);
      const sh = Math.round(H * SCALE);
      const dr = Math.round(ValidationConfig.dotRadius * SCALE);
      logger.log(`0,session_start,-1,-,-,-,-,${sw},${sh},${dr},-`);
      beginPoint(0);
    }, 1000);

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

  // --- Done screen ---
  if (phase === 'done') {
    return (
      <View style={styles.container}>
        <View style={styles.doneContent}>
          <Text style={styles.doneIcon}>✓</Text>
          <Text style={styles.doneTitle}>Validation Recorded</Text>
          <Text style={styles.doneSubtitle}>
            Accuracy will be computed on the PC{'\n'}after offline processing.
          </Text>

          <TouchableOpacity
            style={styles.proceedBtn}
            onPress={() =>
              navigation.navigate('Experiment', { session, videoUri, videoFilename })
            }
          >
            <Text style={styles.proceedBtnText}>Proceed to Experiment</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.recalibrateBtn}
            onPress={() => navigation.navigate('Calibration', { videoUri, videoFilename })}
          >
            <Text style={styles.recalibrateBtnText}>Recalibrate</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // --- Collection screen ---
  const clampedIndex = Math.min(pointIndex, ValidationConfig.pointCount - 1);
  const pos = ValidationConfig.points[clampedIndex];
  const progress = 1.0 - countdown / ValidationConfig.pointDuration;

  return (
    <View style={styles.container}>
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
        Validation {'  '}
        {pointIndex + 1} / {ValidationConfig.pointCount}
      </Text>

      <View
        style={[
          styles.dotWrapper,
          {
            left: pos.x * W - (ValidationConfig.dotRadius + 12),
            top: pos.y * H - (ValidationConfig.dotRadius + 12),
          },
        ]}
      >
        <CalibrationDot
          fixationLetter={fixationLetter}
          progress={progress}
          isCollecting={isCollecting}
          dotRadius={ValidationConfig.dotRadius}
          color="#ffeb3b"
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

  // Done screen
  doneContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
    gap: 16,
  },
  doneIcon: { fontSize: 64, color: '#4caf50' },
  doneTitle: { fontSize: 22, fontWeight: 'bold', color: '#fff' },
  doneSubtitle: {
    fontSize: 15,
    color: 'rgba(255,255,255,0.5)',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 16,
  },
  proceedBtn: {
    backgroundColor: '#4caf50',
    borderRadius: 14,
    padding: 16,
    width: '100%',
    alignItems: 'center',
  },
  proceedBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  recalibrateBtn: { padding: 12 },
  recalibrateBtnText: { color: 'rgba(255,255,255,0.6)', fontSize: 15 },
});
