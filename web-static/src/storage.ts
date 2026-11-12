// localStorage with every access guarded: private windows, blocked site data or a full quota make
// it throw, and the page then works without saving conversations.

export const storage = {
  ok: true,
  get<T>(key: string): T | null {
    try {
      const raw = window.localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : null;
    } catch {
      this.ok = false;
      return null;
    }
  },
  set(key: string, value: unknown): boolean {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch {
      return false;
    }
  },
  probe(): boolean {
    try {
      const k = "ragsys.probe";
      window.localStorage.setItem(k, "1");
      window.localStorage.removeItem(k);
      return true;
    } catch {
      return false;
    }
  },
};
