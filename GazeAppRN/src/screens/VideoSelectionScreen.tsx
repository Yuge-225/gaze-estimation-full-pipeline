import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';
import DocumentPicker from 'react-native-document-picker';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';

type Props = NativeStackScreenProps<RootStackParamList, 'VideoSelection'>;

export default function VideoSelectionScreen({ navigation }: Props) {
  const [selectedUri, setSelectedUri] = useState<string | null>(null);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);

  const pickVideo = async () => {
    try {
      const [file] = await DocumentPicker.pick({
        type: [DocumentPicker.types.video],
        // copyTo caches the file so the URI stays valid after picker closes
        copyTo: 'cachesDirectory',
      });
      setSelectedUri(file.fileCopyUri ?? file.uri);
      setSelectedFilename(file.name ?? 'video.mp4');
    } catch (e) {
      if (!DocumentPicker.isCancel(e)) {
        console.error('[VideoSelection] pick error:', e);
      }
    }
  };

  const handleContinue = () => {
    if (!selectedUri || !selectedFilename) return;
    navigation.navigate('Preview', {
      videoUri: selectedUri,
      videoFilename: selectedFilename,
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.iconText}>▶</Text>

        <View style={styles.textBlock}>
          <Text style={styles.title}>Select Stimulus Video</Text>
          <Text style={styles.subtitle}>
            Choose the video the participant will watch{'\n'}during the experiment.
          </Text>
        </View>

        {selectedFilename ? (
          <View style={styles.selectedRow}>
            <Text style={styles.checkmark}>✓</Text>
            <Text style={styles.filename} numberOfLines={1} ellipsizeMode="middle">
              {selectedFilename}
            </Text>
          </View>
        ) : null}

        <View style={styles.buttons}>
          <TouchableOpacity style={styles.btnBlue} onPress={pickVideo}>
            <Text style={styles.btnText}>
              {selectedFilename ? 'Change Video' : 'Select Video from Library'}
            </Text>
          </TouchableOpacity>

          {selectedUri ? (
            <TouchableOpacity style={styles.btnGreen} onPress={handleContinue}>
              <Text style={styles.btnText}>Continue</Text>
            </TouchableOpacity>
          ) : null}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  content: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
    gap: 24,
  },
  iconText: { fontSize: 64, color: 'rgba(255,255,255,0.3)' },
  textBlock: { alignItems: 'center', gap: 8 },
  title: { fontSize: 22, fontWeight: 'bold', color: '#fff' },
  subtitle: {
    fontSize: 15,
    color: 'rgba(255,255,255,0.5)',
    textAlign: 'center',
    lineHeight: 22,
  },
  selectedRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    gap: 8,
  },
  checkmark: { color: '#4caf50', fontSize: 20 },
  filename: {
    color: 'rgba(255,255,255,0.85)',
    fontSize: 14,
    fontFamily: 'monospace',
    flex: 1,
  },
  buttons: { width: '100%', gap: 14 },
  btnBlue: {
    backgroundColor: '#2196f3',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
  },
  btnGreen: {
    backgroundColor: '#4caf50',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
  },
  btnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
});
