// This function converts linear-scaled frequency decibels from an AnalyserNode's frequncy data to Bark scale [https://en.wikipedia.org/wiki/Bark_scale]
// This implementation uses a simple approach of mapping indices in the linear-scaled array to the closest
// Bark scale center frequency and is not intended to be an accurate representation, but rather "close-enough" for visualization purposes
const barkCenterFrequencies = [
  50, 150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150,
  2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500,
]; // Center frequency value in Hz

// Min/max values from https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode/getByteFrequencyData
const minValue = 0;
const maxValue = 255;

export function convertLinearFrequenciesToBark(
  linearData: Uint8Array,
  sampleRate: number,
): number[] {
  const maxFrequency = sampleRate / 2;
  const frequencyResolution = maxFrequency / linearData.length;

  const barkFrequencies = barkCenterFrequencies.map((barkFreq) => {
    const linearDataIndex = Math.round(barkFreq / frequencyResolution);
    if (linearDataIndex >= 0 && linearDataIndex < linearData.length) {
      return (
        (((linearData[linearDataIndex] ?? 0) - minValue) /
          (maxValue - minValue)) *
        2
      );
    } else {
      return 0;
    }
  });

  return barkFrequencies;
}
