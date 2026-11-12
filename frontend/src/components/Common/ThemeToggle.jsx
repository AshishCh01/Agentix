import { Moon, Sun } from "lucide-react";
import { useTheme } from "../../hooks/useTheme";
import { IconButton } from "../ui/Button";

export const ThemeToggle = ({ className }) => {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";
  return (
    <IconButton
      label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      onClick={toggleTheme}
      className={className}
    >
      {isDark ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
    </IconButton>
  );
};
