export const GazeConfig = {
  bins: 90,
  binWidth: 4.0,
  angle: 180.0,
  inputSize: 448,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
};

export const CalibrationConfig = {
  pointCount: 16,
  pointDuration: 10.0,       // seconds per dot (auto-advances)
  transitionDelay: 0.5,      // seconds between dots
  dotRadius: 18,
  points: [
    // 4 corners
    { x: 0.05, y: 0.05 }, { x: 0.95, y: 0.05 }, { x: 0.05, y: 0.95 }, { x: 0.95, y: 0.95 },
    // 4 edge midpoints
    { x: 0.50, y: 0.05 }, { x: 0.50, y: 0.95 }, { x: 0.05, y: 0.50 }, { x: 0.95, y: 0.50 },
    // 4 inner ring
    { x: 0.25, y: 0.25 }, { x: 0.75, y: 0.25 }, { x: 0.25, y: 0.75 }, { x: 0.75, y: 0.75 },
    // 4 center cluster
    { x: 0.35, y: 0.42 }, { x: 0.65, y: 0.42 }, { x: 0.35, y: 0.58 }, { x: 0.65, y: 0.58 },
  ] as { x: number; y: number }[],
  fixationLetters: 'ACEFHJKLMNPRSTUY'.split(''),
};

export const ValidationConfig = {
  pointCount: 9,
  pointDuration: 10.0,
  transitionDelay: 0.5,
  dotRadius: 18,
  // 3×3 grid at midpoints between the 4×4 calibration grid cells
  points: [
    { x: 0.25, y: 0.22 }, { x: 0.50, y: 0.22 }, { x: 0.75, y: 0.22 },
    { x: 0.25, y: 0.50 }, { x: 0.50, y: 0.50 }, { x: 0.75, y: 0.50 },
    { x: 0.25, y: 0.78 }, { x: 0.50, y: 0.78 }, { x: 0.75, y: 0.78 },
  ] as { x: number; y: number }[],
};

export const CameraConfig = {
  frameRate: 30,
  width: 1920,
  height: 1080,
};

export const AoiWindows: { label: string; start: number; end: number | null }[] = [
  { label: 'Neck',       start: 0,   end: 45  },
  { label: 'Clavicle',   start: 50,  end: 95  },
  { label: 'Upper Arms', start: 101, end: 145 },
  { label: 'Stomach',    start: 151, end: 196 },
  { label: 'Hips',       start: 202, end: 247 },
  { label: 'Thighs',     start: 253, end: null },
];
