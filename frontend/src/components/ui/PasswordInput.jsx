import { useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { Input } from "./Field";

export const PasswordInput = (props) => {
  const [visible, setVisible] = useState(false);
  return (
    <div className="relative">
      <Input {...props} type={visible ? "text" : "password"} className="pr-11" />
      <button
        type="button"
        onClick={() => setVisible((v) => !v)}
        aria-label={visible ? "Hide password" : "Show password"}
        className="absolute right-1 top-1 flex size-9 items-center justify-center rounded-md text-ink-3 hover:text-ink"
      >
        {visible ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
      </button>
    </div>
  );
};
