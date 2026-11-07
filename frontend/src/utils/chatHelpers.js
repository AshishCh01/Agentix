/**
 * Pure helper functions for chat message processing, stream event handling,
 * not-covered canned response detection, and verification badge configuration.
 */

/**
 * Checks whether an agent node is responsible for generating answer text to display to the user.
 *
 * @param {string|null} nodeName
 * @returns {boolean}
 */
export const isAnswerNode = (nodeName) => {
  return nodeName === "answer" || nodeName === "greeting" || nodeName === "direct_answer";
};

/**
 * Resets the in-progress assistant draft content when a new answer-producing node begins.
 * This guarantees retried drafts replace earlier rejected drafts rather than concatenating.
 *
 * @param {Array} messages - List of chat messages
 * @param {string} nodeName - The node that just started
 * @returns {Array} Updated messages array
 */
export const updateMessagesOnNodeStart = (messages, nodeName) => {
  if (!isAnswerNode(nodeName) || !messages || messages.length === 0) {
    return messages;
  }
  const updated = [...messages];
  const lastIndex = updated.length - 1;
  const lastMsg = updated[lastIndex];
  if (lastMsg && lastMsg.role === "assistant" && lastMsg.isStreaming) {
    updated[lastIndex] = {
      ...lastMsg,
      content: "",
    };
  }
  return updated;
};

/**
 * Appends streamed tokens to the active assistant message, discarding tokens from non-answer nodes
 * (e.g. structured reflection evaluations or supervisor query rewrites).
 *
 * @param {Array} messages - List of chat messages
 * @param {string} tokenContent - Token text chunk
 * @param {string|null} currentNode - Current active node name
 * @returns {Array} Updated messages array
 */
export const updateMessagesOnToken = (messages, tokenContent, currentNode) => {
  if (!tokenContent || !messages || messages.length === 0) {
    return messages;
  }
  if (currentNode && !isAnswerNode(currentNode)) {
    return messages;
  }
  const updated = [...messages];
  const lastIndex = updated.length - 1;
  const lastMsg = updated[lastIndex];
  if (lastMsg && lastMsg.role === "assistant") {
    updated[lastIndex] = {
      ...lastMsg,
      content: (lastMsg.content || "") + tokenContent,
    };
  }
  return updated;
};

export const NOT_COVERED_PROMPT_SNIPPET =
  "I couldn't find this in your uploaded documents. Would you like me to search the web instead?";

/**
 * Detects whether an assistant response is the canned 'not covered' reply
 * prompting for optional web search fallback.
 *
 * @param {string|null} content - Assistant message text
 * @returns {boolean}
 */
export const isNotCoveredReply = (content) => {
  if (typeof content !== "string" || !content) return false;
  return (
    content.includes(NOT_COVERED_PROMPT_SNIPPET) ||
    (content.includes("I couldn't find this in your uploaded documents") &&
      content.includes("Would you like me to search the web instead?"))
  );
};

/**
 * Checks whether an assistant message was answered from web search sources.
 *
 * @param {Object} message - Chat message object
 * @returns {boolean}
 */
export const isWebSourceMessage = (message) => {
  if (!message) return false;
  if (message.context_source === "web" || message.intent === "WEB_SEARCH") {
    return true;
  }
  if (Array.isArray(message.sources) && message.sources.length > 0) {
    return message.sources.some(
      (src) => Boolean(src.url) || src.source_type === "web"
    );
  }
  return false;
};

/**
 * Determines the verification badge configuration for an assistant message.
 * Returns null if the message is unverified, older/history, web answer, or not-covered.
 *
 * @param {Object} message - Chat message object
 * @returns {{ type: "verified"|"unverified", label: string, title: string, ariaLabel: string, isWeb: boolean } | null}
 */
export const getVerificationBadgeConfig = (message) => {
  if (!message || message.role !== "assistant" || message.isStreaming) {
    return null;
  }

  // Do not show a badge on the not-covered reply
  if (isNotCoveredReply(message.content)) {
    return null;
  }

  const isWeb = isWebSourceMessage(message);

  if (message.verified === true) {
    const label = isWeb ? "Verified against web sources" : "Verified against your documents";
    return {
      type: "verified",
      label,
      title: label,
      ariaLabel: label,
      isWeb,
    };
  }

  if (message.verified === false) {
    return {
      type: "unverified",
      label: "Could not be fully verified",
      title: "Could not be fully verified",
      ariaLabel: "Could not be fully verified",
      isWeb,
    };
  }

  return null;
};

/**
 * Extracts final reflection verdict and verification status from a completion event.
 * Ensures that the final verdict from the last reflection pass is preserved.
 *
 * @param {Object} event - SSE completion event payload
 * @returns {{ reflection: Object|null, verified: boolean|null, reflection_verdict: string|null }}
 */
export const extractFinalReflection = (event) => {
  if (!event) {
    return { reflection: null, verified: null, reflection_verdict: null };
  }

  const verified =
    event.verified !== undefined && event.verified !== null
      ? Boolean(event.verified)
      : event.reflection?.verified !== undefined && event.reflection?.verified !== null
        ? Boolean(event.reflection.verified)
        : null;

  const reflection_verdict =
    event.reflection_verdict ||
    (event.reflection?.result?.is_grounded && event.reflection?.result?.is_relevant
      ? "passed"
      : event.reflection?.verified === false
        ? "failed"
        : null);

  return {
    reflection: event.reflection || null,
    verified,
    reflection_verdict,
  };
};
