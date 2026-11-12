import { MAX_UPLOAD_BYTES } from "./constants";

export const MAX_UPLOAD_MB = Math.floor(MAX_UPLOAD_BYTES / (1024 * 1024));
export const VALID_EXTENSIONS = ["pdf", "docx", "csv", "xlsx", "txt", "md", "json"];
export const ACCEPT_ATTR = `${VALID_EXTENSIONS.map((e) => `.${e}`).join(",")},image/*`;
export const SUPPORTED_LABEL = "PDF, Word, Excel, CSV, text, Markdown, JSON and images";

export const isValidFile = (file) => {
  const ext = file.name.split(".").pop().toLowerCase();
  return file.type.startsWith("image/") || VALID_EXTENSIONS.includes(ext);
};

/** Returns an error string, or null when the file is acceptable. */
export const validateFile = (file) => {
  if (!isValidFile(file)) return `Unsupported file type. Use ${SUPPORTED_LABEL}.`;
  if (file.size > MAX_UPLOAD_BYTES) {
    return `That file is too large. The limit is ${MAX_UPLOAD_MB} MB.`;
  }
  return null;
};

export const formatFileSize = (bytes) => {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1);
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};
