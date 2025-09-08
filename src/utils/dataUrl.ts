// dataUrl.ts - Utility for handling data URLs
export const isDataUrl = (url) => url.startsWith('data:');

export const parseDataUrl = (url) => {
  const match = /^data:([^;]+);base64,(.+)$/.exec(url);
  return match ? { mediaType: match[1], data: match[2] } : null;
};
