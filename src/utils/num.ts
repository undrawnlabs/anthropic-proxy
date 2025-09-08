// num.ts - Utility for number parsing
export const num = (value, defaultValue) => {
  return Number.isFinite(Number(value)) ? Number(value) : defaultValue;
};
