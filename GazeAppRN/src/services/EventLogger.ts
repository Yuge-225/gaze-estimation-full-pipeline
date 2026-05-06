import RNFS from 'react-native-fs';

export class EventLogger {
  private lines: string[] = [];
  private startTime = 0;
  private filePath: string;

  constructor(filePath: string, header: string) {
    this.filePath = filePath;
    this.lines.push(header);
  }

  markStart(): void {
    this.startTime = Date.now();
  }

  elapsedMs(): number {
    return Date.now() - this.startTime;
  }

  log(row: string): void {
    this.lines.push(row);
  }

  async save(): Promise<void> {
    try {
      await RNFS.writeFile(this.filePath, this.lines.join('\n'), 'utf8');
    } catch (e) {
      console.error('[EventLogger] save failed:', e);
    }
  }
}
