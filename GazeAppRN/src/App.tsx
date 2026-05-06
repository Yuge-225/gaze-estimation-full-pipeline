import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { CalibrationSession } from './models/CalibrationSession';

import VideoSelectionScreen from './screens/VideoSelectionScreen';
import PreviewScreen from './screens/PreviewScreen';
import CalibrationScreen from './screens/CalibrationScreen';
import ValidationScreen from './screens/ValidationScreen';
import ExperimentScreen from './screens/ExperimentScreen';
import ExportScreen from './screens/ExportScreen';

export type RootStackParamList = {
  VideoSelection: undefined;
  Preview: { videoUri: string; videoFilename: string };
  Calibration: { videoUri: string; videoFilename: string };
  Validation: { session: CalibrationSession; videoUri: string; videoFilename: string };
  Experiment: { session: CalibrationSession; videoUri: string; videoFilename: string };
  Export: { session: CalibrationSession };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="VideoSelection"
        screenOptions={{ headerShown: false, animation: 'none' }}
      >
        <Stack.Screen name="VideoSelection" component={VideoSelectionScreen} />
        <Stack.Screen name="Preview" component={PreviewScreen} />
        <Stack.Screen name="Calibration" component={CalibrationScreen} />
        <Stack.Screen name="Validation" component={ValidationScreen} />
        <Stack.Screen name="Experiment" component={ExperimentScreen} />
        <Stack.Screen name="Export" component={ExportScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
