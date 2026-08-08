export async function* parseSSEStream(reader) {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    
    // Keep incomplete last line in the buffer
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data:")) {
        const jsonStr = trimmed.replace(/^data:\s*/, "");
        if (!jsonStr) continue;
        try {
          yield JSON.parse(jsonStr);
        } catch (e) {
          console.warn("Failed to parse SSE JSON chunk:", jsonStr, e);
        }
      }
    }
  }

  // Flush remaining buffer
  if (buffer.trim().startsWith("data:")) {
    const jsonStr = buffer.trim().replace(/^data:\s*/, "");
    if (jsonStr) {
      try {
        yield JSON.parse(jsonStr);
      } catch (e) {
        console.warn("Failed to parse trailing SSE JSON chunk:", jsonStr, e);
      }
    }
  }
}