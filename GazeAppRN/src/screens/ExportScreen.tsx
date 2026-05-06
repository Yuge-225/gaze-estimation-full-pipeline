import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import RNFS from 'react-native-fs';
import Share from 'react-native-share';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import type { RootStackParamList } from '../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Export'>;

interface FileItem {
  name: string;
  path: string;
  size: string;
}

export default function ExportScreen({ navigation, route }: Props) {
  const { session } = route.params;
  const [files, setFiles] = useState<FileItem[]>([]);

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      const items = await RNFS.readDir(session.directory);
      const sorted = items
        .filter(i => i.isFile())
        .sort((a, b) => a.name.localeCompare(b.name))
        .map(i => ({
          name: i.name,
          path: i.path,
          size: formatBytes(Number(i.size)),
        }));
      setFiles(sorted);
    } catch (e) {
      console.error('[Export] readDir error:', e);
    }
  };

  const handleShare = async () => {
    try {
      await Share.open({
        urls: files.map(f => `file://${f.path}`),
        title: 'GazeApp Session Data',
        // On Android, Share opens the native share sheet (email, Bluetooth, Drive, etc.)
        // On iOS, this opens the UIActivityViewController
      });
    } catch (e) {
      // User cancelled — ignore
    }
  };

  const dirName = session.directory.split('/').pop() ?? '';

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Header */}
        <Text style={styles.successIcon}>✓</Text>
        <Text style={styles.title}>Session Complete</Text>
        <Text style={styles.dirName}>{dirName}</Text>

        {/* File list */}
        <View style={styles.fileList}>
          {files.map((file, i) => (
            <View
              key={file.name}
              style={[styles.fileRow, i > 0 && styles.fileRowBorder]}
            >
              <Text style={[styles.fileIcon, isVideo(file.name) && styles.fileIconVideo]}>
                {fileEmoji(file.name)}
              </Text>
              <Text
                style={[styles.fileName, isVideo(file.name) && styles.fileNameVideo]}
                numberOfLines={1}
              >
                {file.name}
              </Text>
              <Text style={styles.fileSize}>{file.size}</Text>
            </View>
          ))}
        </View>

        {/* Actions */}
        <TouchableOpacity style={styles.shareBtn} onPress={handleShare}>
          <Text style={styles.shareBtnText}>Share / Export Files</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.newSessionBtn}
          onPress={() => navigation.navigate('VideoSelection')}
        >
          <Text style={styles.newSessionText}>New Session</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

function isVideo(name: string): boolean {
  return name.toLowerCase().endsWith('.mp4');
}

function fileEmoji(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase();
  if (ext === 'mp4') return '▶';
  if (ext === 'csv') return '⊞';
  if (ext === 'json') return '{}';
  return '□';
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  content: {
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingTop: 48,
    paddingBottom: 48,
    gap: 16,
  },
  successIcon: { fontSize: 52, color: '#4caf50' },
  title: { fontSize: 22, fontWeight: 'bold', color: '#fff' },
  dirName: {
    fontSize: 11,
    color: 'rgba(255,255,255,0.4)',
    fontFamily: 'monospace',
    marginBottom: 16,
  },
  fileList: {
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 12,
    marginBottom: 8,
  },
  fileRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    gap: 10,
  },
  fileRowBorder: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255,255,255,0.08)',
  },
  fileIcon: {
    width: 24,
    textAlign: 'center',
    color: 'rgba(255,255,255,0.5)',
    fontSize: 14,
  },
  fileIconVideo: { color: '#4dd0e1' },
  fileName: {
    flex: 1,
    color: 'rgba(255,255,255,0.85)',
    fontSize: 13,
    fontFamily: 'monospace',
  },
  fileNameVideo: { color: '#4dd0e1' },
  fileSize: {
    color: 'rgba(255,255,255,0.4)',
    fontSize: 12,
    fontFamily: 'monospace',
  },
  shareBtn: {
    backgroundColor: '#2196f3',
    borderRadius: 14,
    padding: 16,
    width: '100%',
    alignItems: 'center',
    marginTop: 8,
  },
  shareBtnText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  newSessionBtn: { padding: 12 },
  newSessionText: { color: 'rgba(255,255,255,0.55)', fontSize: 15 },
});
