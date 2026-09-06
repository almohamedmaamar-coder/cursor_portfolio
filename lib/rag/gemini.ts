/**
 * lib/rag/gemini.ts
 * Direct native client for Google Gemini API.
 * Uses standard fetch — zero external dependencies, zero LangChain version mismatches.
 */

export async function callGemini(
  prompt: string,
  systemInstruction?: string,
  temperature = 0.1
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error("Missing GEMINI_API_KEY environment variable");
  }

  const model = process.env.GOOGLE_MODEL || "gemini-2.5-flash";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const body: {
    contents: Array<{ role: string; parts: Array<{ text: string }> }>;
    generationConfig: { temperature: number };
    systemInstruction?: { parts: Array<{ text: string }> };
  } = {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: { temperature },
  };

  if (systemInstruction) {
    body.systemInstruction = { parts: [{ text: systemInstruction }] };
  }

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Gemini API error ${res.status}: ${errText}`);
  }

  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
}