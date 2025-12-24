// API configuration
//const API_BASE_URL = "https://improved-adventure-jjvg6jwgqwpvc5v45-8000.app.github.dev";
const API_BASE_URL = "http://localhost:8000";
export const api = {
  baseUrl: API_BASE_URL,
  endpoints: {
    documents: `${API_BASE_URL}/api/documents`,
    chat: `${API_BASE_URL}/api/chat`,
  },
};