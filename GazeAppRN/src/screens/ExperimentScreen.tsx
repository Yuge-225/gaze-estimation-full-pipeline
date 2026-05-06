import React, { useRef, useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Camera, useCameraDevice, useCameraFormat } from 'react-native-vision-camera';
import Video from 'react-native-video';
import RNFS from 'react-native-fs';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';
import { CameraConfig, AoiWindows } from '../constants/config';
import { EventLogger } from '../services/EventLogger';

type Props = NativeStackScreenProps<RootStackParamList, 'Experiment'>;

export default function ExperimentScreen({ navigation, route }: Props) {
  const { session, videoUri, videoFilename } = route.params;
  const device = useCameraDevice('front');
  const format = useCameraFormat(device, [{ fps: CameraConfig.frameRate }]);
  const cameraRef = useRef<Camera>(null);
  const loggerRef = useRef<EventLogger | null>(null);
  const isRecording = useRef(false);
  const videoDurationRef = useRef(0);

  // 'ready' → user sees confirm screen; 'recording' → fullscreen video plays
  const [phase, setPhase] = useState<'ready' | 'recording'>('ready');
  const [paused, setPaused] = useState(true);
  const [currentAoi, setCurrentAoi] = useState<string | null>(null);

  const startExperiment = async () => {
    // Switch to recording phase immediately so the Video component mounts
    setPhase('recording');

    // Brief pause to let camera and video finish mounting
    await new Promise(r => setTimeout(r, 500));

    const logger = new EventLogger(
      session.experimentEventsPath,
      'elapsed_ms,event_type,stimulus_filename,stimulus_time_ms,stimulus_duration_ms',
    );
    loggerRef.current = logger;
    logger.markStart();
    logger.log(`0,session_start,${videoFilename},-,-`);

    isRecording.current = true;
    await cameraRef.current?.startRecording({
      fileType: 'mp4',
      onRecordingFinished: async video => {
        try {
          await RNFS.copyFile(video.path, session.experimentVideoPath);
        } catch {}
      },
      onRecordingError: e => console.error('[Experiment] recording error:', e),
    });

    // Start video playback
    setPaused(false);
    const durMs = Math.round(videoDurationRef.current * 1000);
    logger.log(`${logger.elapsedMs()},stimulus_start,${videoFilename},0,${durMs}`);
  };

  const handleVideoEnd = async () => {
    const logger = loggerRef.current;
    const ms = logger?.elapsedMs() ?? 0;
    const durMs = Math.round(videoDurationRef.current * 1000);
    logger?.log(`${ms},stimulus_end,${videoFilename},${durMs},${durMs}`);
    logger?.log(`${ms},session_end,${videoFilename},-,-`);

    if (isRecording.current) {
      isRecording.current = false;
      await cameraRef.current?.stopRecording();
    }
    await logger?.save();

    navigation.navigate('Export', { session });
  };

  useEffect(() => {
    return () => {
      if (isRecording.current) {
        isRecording.current = false;
        cameraRef.current?.stopRecording();
      }
      loggerRef.current?.save();
    };
  }, []);

  return (
    <View style={styles.container}>
      {/* Hidden front camera — records throughout the experiment */}
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

      {/* Ready screen */}
      {phase === 'ready' && (
        <View style={styles.readyOverlay}>
          <Text style={styles.readyIcon}>▶</Text>
          <Text style={styles.readyTitle}>Ready</Text>
          <Text style={styles.readyFilename} numberOfLines={2}>
            {videoFilename}
          </Text>
          <Text style={styles.readyHint}>
            The stimulus video will play fullscreen.{'\n'}
            The front camera will record throughout.
          </Text>
          <TouchableOpacity style={styles.startBtn} onPress={startExperiment}>
            <Text style={styles.startBtnText}>Start Recording</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Stimulus video — mounted early so it pre-buffers during ready phase */}
      <Video
        source={{ uri: videoUri }}
        style={phase === 'recording' ? StyleSheet.absoluteFill : styles.hiddenVideo}
        resizeMode="contain"
        paused={paused}
        onLoad={data => {
          videoDurationRef.current = data.duration;
        }}
        onProgress={({ currentTime }) => {
          const aoi = AoiWindows.find(
            w => currentTime >= w.start && (w.end === null || currentTime <= w.end),
          )?.label ?? null;
          setCurrentAoi(aoi);
        }}
        onEnd={handleVideoEnd}
        controls={false}
        ignoreSilentSwitch="ignore"
        playInBackground={false}
      />

      {/* REC badge — only visible during recording */}
      {phase === 'recording' && (
        <View style={styles.recBadge} pointerEvents="none">
          <View style={styles.recDot} />
          <Text style={styles.recText}>REC</Text>
        </View>
      )}

      {/* AOI badge — shows current area of interest */}
      {phase === 'recording' && currentAoi !== null && (
        <View style={styles.aoiBadge} pointerEvents="none">
          <Text style={styles.aoiLabel}>AOI</Text>
          <Text style={styles.aoiText}>{currentAoi}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  hiddenCamera: { width: 1, height: 1, position: 'absolute', opacity: 0 },
  hiddenVideo: { width: 0, height: 0 },

  // Ready screen
  readyOverlay: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
    gap: 16,
  },
  readyIcon: { fontSize: 56, color: 'rgba(255,255,255,0.4)' },
  readyTitle: { fontSize: 22, fontWeight: 'bold', color: '#fff' },
  readyFilename: {
    fontSize: 15,
    color: '#fff',
    textAlign: 'center',
    fontFamily: 'monospace',
  },
  readyHint: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.45)',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 8,
  },
  startBtn: {
    backgroundColor: '#4caf50',
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 48,
  },
  startBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },

  // REC indicator
  recBadge: {
    position: 'absolute',
    top: 56,
    right: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(0,0,0,0.45)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  recDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#f44336',
  },
  recText: { color: '#fff', fontSize: 11, fontWeight: '600' },

  aoiBadge: {
    position: 'absolute',
    top: 96,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    alignItems: 'center',
  },
  aoiLabel: { color: 'rgba(255,255,255,0.5)', fontSize: 9, fontWeight: '600', letterSpacing: 1 },
  aoiText: { color: '#fff', fontSize: 12, fontWeight: '600' },
});
