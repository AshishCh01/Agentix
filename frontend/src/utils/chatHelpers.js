/**
 * Pure helper functions for chat message processing, stream event handling,
 * not-covered canned response detection, and verification badge configuration.
 */

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
