import RNFS from 'react-native-fs';
import { Dimensions, Platform } from 'react-native';
import { CameraConfig } from '../constants/config';

export interface CalibrationSession {
  sessionId: string;
  directory: string;
  calibrationVideoPath: string;
  calibrationEventsPath: string;
  validationVideoPath: string;
  validationEventsPath: string;
  experimentVideoPath: string;
  experimentEventsPath: string;
  metadataPath: string;
}

export async function createNewSession(): Promise<CalibrationSession> {
  const sessionsDir = `${RNFS.DocumentDirectoryPath}/Sessions`;

  if (!(await RNFS.exists(sessionsDir))) {
    await RNFS.mkdir(sessionsDir);
  }

  const now = new Date();
  const pad = (n: number) => n.toString().padStart(2, '0');
  const dateStr =
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}` +
    `_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  const uid = Math.random().toString(36).substring(2, 10).toUpperCase();
  const sessionDir = `${sessionsDir}/session_${dateStr}_${uid}`;

  await RNFS.mkdir(sessionDir);

  const session: CalibrationSession = {
    sessionId: uid,
    directory: sessionDir,
    calibrationVideoPath: `${sessionDir}/calibration.mp4`,
    calibrationEventsPath: `${sessionDir}/calibration_events.csv`,
    validationVideoPath: `${sessionDir}/validation.mp4`,
    validationEventsPath: `${sessionDir}/validation_events.csv`,
    experimentVideoPath: `${sessionDir}/experiment.mp4`,
    experimentEventsPath: `${sessionDir}/experiment_events.csv`,
    metadataPath: `${sessionDir}/metadata.json`,
  };

  await writeMetadata(session);
  return session;
}

async function writeMetadata(session: CalibrationSession): Promise<void> {
  const dim = Dimensions.get('screen');
  const meta = {
    session_id: session.sessionId,
    platform: Platform.OS,
    os_version: Platform.Version.toString(),
    camera_resolution: `${CameraConfig.width}x${CameraConfig.height}`,
    camera_fps: CameraConfig.frameRate,
    screen_width_pt: dim.width,
    screen_height_pt: dim.height,
    screen_scale: dim.scale,
    screen_width_px: Math.round(dim.width * dim.scale),
    screen_height_px: Math.round(dim.height * dim.scale),
    created_at: new Date().toISOString(),
  };
  await RNFS.writeFile(session.metadataPath, JSON.stringify(meta, null, 2), 'utf8');
}
