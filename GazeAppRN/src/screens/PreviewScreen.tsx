import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
} from 'react-native';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';
import { CameraConfig } from '../constants/config';

type Props = NativeStackScreenProps<RootStackParamList, 'Preview'>;


export default function PreviewScreen({ navigation, route }: Props) {
  const { videoUri, videoFilename } = route.params;
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const [cameraActive, setCameraActive] = useState(false);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission]);

  useEffect(() => {
    const t = setTimeout(() => setCameraActive(true), 150);
    return () => clearTimeout(t);
  }, []);


  return (
    <View style={styles.container}>
      {device && hasPermission && (
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={cameraActive}
        />
      )}

      <View style={styles.topOverlay}>
        <Text style={styles.mainTitle}>Gaze Tracking Study</Text>
        <Text style={styles.cameraInfo}>Front camera · {CameraConfig.frameRate} fps</Text>
      </View>

      <View style={styles.bottomOverlay}>
        <View style={styles.statusPanel}>
          <Text style={styles.statusText}>
            Position your face in the center of the screen,{'\n'}
            30–60 cm away, then tap Start.
          </Text>
        </View>

        <TouchableOpacity
          style={styles.startBtn}
          onPress={() => navigation.navigate('Calibration', { videoUri, videoFilename })}
        >
          <Text style={styles.startBtnText}>Start Calibration</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  topOverlay: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  mainTitle: { fontSize: 20, fontWeight: 'bold', color: '#fff' },
  cameraInfo: { fontSize: 12, color: 'rgba(255,255,255,0.45)', marginTop: 4 },
  bottomOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 16,
    gap: 12,
  },
  statusPanel: {
    backgroundColor: 'rgba(20,20,20,0.85)',
    borderRadius: 12,
    padding: 16,
  },
  statusText: { color: 'rgba(255,255,255,0.75)', fontSize: 14, lineHeight: 22, textAlign: 'center' },
  startBtn: {
    backgroundColor: '#4caf50',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 16,
  },
  startBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
