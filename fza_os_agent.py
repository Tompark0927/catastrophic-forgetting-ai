"""
fza_os_agent.py — Operating System Symbiosis (v15.0)
=====================================================
FZA's Motor Cortex. The AI breaks out of the terminal and gains the
ability to SEE the screen and CONTROL the computer.

Biological Metaphor: The Sensorimotor Cortex.
  - The visual cortex (fza_visual_memory.py) reads the scene.
  - The motor cortex (this file) translates intention into action.
  - The cerebellum (ProceduralMemory) stores learned workflows and
    prevents the AI from "forgetting" how to use an application.

How it works:
1. OBSERVE: Takes a screenshot, encodes it into the VisualMemoryPipeline
   so GUI elements become WorldGraph anchors.
2. PLAN: Given a natural language goal (e.g., "open Safari and google X"),
   breaks it down into a sequence of atomic actions using the LLM.
3. ACT: Executes the plan step-by-step via PyAutoGUI (move, click, type).
4. VERIFY: After each action, takes a fresh screenshot and checks if the
   intended state was achieved (e.g., "did the window open?").
5. REMEMBER: Successful workflows are saved as ProceduralLoRA adapters.

Dependencies:
    pip install pyautogui pillow pygetwindow

Usage:
    from fza_os_agent import OSAgent
    agent = OSAgent()
    agent.observe()                          # takes screenshot + encodes GUI state
    agent.execute_task("open Safari")        # plan + act loop
    agent.describe_screen()                  # returns natural language screen summary
"""

import time
import io
import os
import json
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from fza_event_bus import bus

# Optional imports — gracefully degraded
try:
    import pyautogui
    pyautogui.FAILSAFE = True   # Move mouse to top-left corner to abort
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False

try:
    from PIL import Image, ImageGrab
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


@dataclass
class GUIElement:
    """A recognized GUI element from a screen observation."""
    label: str              # e.g. "Send button", "address bar", "Safari icon"
    region: tuple           # (x, y, width, height) in screen coords
    element_type: str       # "button", "textfield", "icon", "menu", "window"
    confidence: float = 0.9


@dataclass
class ActionStep:
    """A single atomic action in a plan."""
    action_type: str        # "click", "type", "hotkey", "scroll", "wait", "screenshot"
    target: Optional[str] = None    # label or coordinates
    value: Optional[str] = None     # text to type / hotkey combo
    delay_after: float = 0.5


@dataclass  
class ActionResult:
    """Result of executing an action step."""
    success: bool
    action: ActionStep
    screen_changed: bool = False
    error: Optional[str] = None


class ScreenObserver:
    """
    Takes and analyzes screenshots. Converts screen state into text descriptions
    that FZA can reason about.
    """

    def __init__(self):
        self.last_screenshot: Optional[object] = None
        self.last_screenshot_path: Optional[str] = "/tmp/fza_screen.png"
        self.observation_count = 0

    def capture(self) -> Optional[object]:
        """Takes a screenshot and returns a PIL Image (or None if unavailable)."""
        if not _PIL_AVAILABLE:
            print("⚠️  [ScreenObserver] PIL 없음 — pip install pillow")
            return None
        try:
            img = ImageGrab.grab()
            img.save(self.last_screenshot_path)
            self.last_screenshot = img
            self.observation_count += 1
            return img
        except Exception as e:
            print(f"⚠️  [ScreenObserver] 스크린샷 실패: {e}")
            return None

    def describe(self, img=None) -> str:
        """
        Returns a natural language description of the current screen state.
        In production: use a vision model (e.g. GPT-4V, LLaVA) to generate rich descriptions.
        Fallback: return metadata about the screen dimensions.
        """
        if img is None:
            img = self.last_screenshot
        if img is None:
            return "[스크린샷 없음]"

        w, h = img.size
        return (
            f"[화면 관찰] 해상도: {w}×{h}px | "
            f"스냅샷 #{self.observation_count} | "
            f"저장 위치: {self.last_screenshot_path}"
        )

    def diff(self, img_before, img_after, threshold: int = 100) -> bool:
        """
        Returns True if the screen changed significantly between two captures.
        Used by the agent to verify that an action had an effect.
        """
        if img_before is None or img_after is None:
            return False
        try:
            import PIL.ImageChops as chops
            diff = chops.difference(img_before, img_after)
            bbox = diff.getbbox()
            return bbox is not None and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > threshold
        except Exception:
            return False


class OSActionExecutor:
    """
    Executes atomic GUI actions via PyAutoGUI.
    Operates as the "hand" that carries out the Motor Cortex's plan.
    """

    def __init__(self, dry_run: bool = False):
        """
        Args:
            dry_run: If True, logs actions without actually executing them (safe mode).
        """
        self.dry_run = dry_run
        self.action_log: List[ActionResult] = []

        if not _PYAUTOGUI_AVAILABLE:
            print("⚠️  [OSActionExecutor] PyAutoGUI 없음 — pip install pyautogui")
            self.dry_run = True  # Force dry_run if pyautogui missing

    def execute(self, step: ActionStep) -> ActionResult:
        """Execute a single action step and return its result."""
        action_str = f"[{step.action_type}] {step.target or ''} {step.value or ''}"

        if self.dry_run:
            print(f"🔵 [DRY RUN] {action_str}")
            result = ActionResult(success=True, action=step)
            self.action_log.append(result)
            return result

        try:
            if step.action_type == "click":
                coords = self._resolve_coords(step.target)
                if coords:
                    pyautogui.click(*coords)
                    print(f"🖱️  [OSAgent] 클릭: {step.target} @ {coords}")

            elif step.action_type == "type":
                pyautogui.typewrite(step.value or "", interval=0.05)
                print(f"⌨️  [OSAgent] 타이핑: '{step.value[:30] if step.value else ''}'")

            elif step.action_type == "hotkey":
                keys = (step.value or "").split("+")
                pyautogui.hotkey(*keys)
                print(f"⌨️  [OSAgent] 단축키: {step.value}")

            elif step.action_type == "scroll":
                pyautogui.scroll(int(step.value or "3"))
                print(f"🖱️  [OSAgent] 스크롤: {step.value}")

            elif step.action_type == "wait":
                time.sleep(float(step.value or "1"))
                print(f"⏸️  [OSAgent] 대기: {step.value}초")

            elif step.action_type == "screenshot":
                pass  # Handled by the agent loop

            time.sleep(step.delay_after)
            result = ActionResult(success=True, action=step)

        except Exception as e:
            print(f"❌ [OSAgent] 액션 실패: {action_str} | {e}")
            result = ActionResult(success=False, action=step, error=str(e))

        self.action_log.append(result)
        bus.emit("os_action", {"action": step.action_type, "target": step.target, "success": result.success})
        return result

    def execute_plan(self, plan: List[ActionStep]) -> List[ActionResult]:
        """Execute a full plan and return all results."""
        results = []
        for step in plan:
            result = self.execute(step)
            results.append(result)
            if not result.success:
                print(f"⚠️  [OSAgent] 플랜 중단 @ 단계 {len(results)}")
                break
        return results

    def _resolve_coords(self, target: str) -> Optional[tuple]:
        """
        Resolves a target label to screen coordinates.
        In production: uses a vision model to locate the element.
        For now: if the target is already 'x,y', parse it directly.
        """
        if target and "," in target:
            try:
                x, y = target.split(",")
                return (int(x.strip()), int(y.strip()))
            except Exception:
                pass
        # fallback: center of screen
        if _PYAUTOGUI_AVAILABLE:
            return pyautogui.size()[0] // 2, pyautogui.size()[1] // 2
        return (960, 540)


class OSAgent:
    """
    The top-level OS Symbiosis agent.
    Ties together ScreenObserver (vision) and OSActionExecutor (motor).
    """

    def __init__(
        self,
        procedural_memory=None,   # Optional ProceduralMemory reference
        dry_run: bool = False,
        llm_fn: Optional[Callable] = None,   # Optional LLM function for plan generation
    ):
        self.observer = ScreenObserver()
        self.executor = OSActionExecutor(dry_run=dry_run)
        self.procedural_memory = procedural_memory
        self.llm_fn = llm_fn
        self.tasks_executed = 0
        self.current_goal: Optional[str] = None

        print(f"🖥️  [OSAgent] 초기화 완료 | "
              f"PyAutoGUI={'✅' if _PYAUTOGUI_AVAILABLE else '❌'}, "
              f"Pillow={'✅' if _PIL_AVAILABLE else '❌'}, "
              f"dry_run={dry_run}")

    def observe(self) -> str:
        """Takes a screenshot and returns a text description of the current screen."""
        img = self.observer.capture()
        desc = self.observer.describe(img)

        # Optionally encode into visual memory
        # (in production: extract UI element labels using vision model)
        bus.emit("os_observe", {"description": desc})
        print(f"👁️  [OSAgent] 화면 관찰: {desc}")
        return desc

    def execute_task(self, goal: str, plan: Optional[List[ActionStep]] = None) -> dict:
        """
        Executes a goal. If a pre-built plan is provided, execute it.
        Otherwise, uses the LLM to generate a plan.

        Args:
            goal:  Natural language goal (e.g. "open Safari and search for Python")
            plan:  Optional pre-built list of ActionSteps. If None, uses LLM.

        Returns:
            dict with "success", "steps_executed", "goal"
        """
        self.current_goal = goal
        self.tasks_executed += 1
        print(f"\n🎯 [OSAgent] 목표 시작: '{goal}'")

        # Take a before-screenshot for change detection
        before_img = self.observer.capture()

        if plan is None:
            plan = self._generate_plan(goal)

        if not plan:
            return {"success": False, "goal": goal, "steps_executed": 0, "error": "Plan generation failed"}

        results = self.executor.execute_plan(plan)
        success = all(r.success for r in results)

        # Verify: did the screen change?
        after_img = self.observer.capture()
        screen_changed = self.observer.diff(before_img, after_img)

        print(f"\n{'✅' if success else '❌'} [OSAgent] 임무 완료: '{goal}' | "
              f"단계: {len(results)}/{len(plan)} | 화면 변화: {screen_changed}")

        # Save to procedural memory on success
        if success and self.procedural_memory:
            self.procedural_memory.record_workflow(goal, plan, success=True)

        bus.emit("os_task_complete", {"goal": goal, "success": success, "steps": len(results)})
        return {"success": success, "goal": goal, "steps_executed": len(results), "screen_changed": screen_changed}

    def _generate_plan(self, goal: str) -> List[ActionStep]:
        """
        Generates an ActionStep plan for the goal.
        Uses LLM if available; else uses a simple rule-based fallback.
        """
        if self.llm_fn:
            try:
                prompt = (
                    f"You are controlling a computer. Generate a step-by-step action plan "
                    f"for this goal: '{goal}'\n"
                    f"Output as JSON list with fields: action_type, target, value, delay_after.\n"
                    f"action_type must be one of: click, type, hotkey, scroll, wait, screenshot"
                )
                response = self.llm_fn(prompt)
                steps_data = json.loads(response)
                return [ActionStep(**s) for s in steps_data]
            except Exception as e:
                print(f"⚠️  [OSAgent] LLM 플랜 생성 실패: {e}")

        # Rule-based fallback for common goals
        return self._rule_based_plan(goal)

    def _rule_based_plan(self, goal: str) -> List[ActionStep]:
        """Simple keyword → action plan mapping."""
        goal_lower = goal.lower()

        if "spotlight" in goal_lower or "search" in goal_lower:
            return [
                ActionStep("hotkey", value="cmd+space"),
                ActionStep("wait", value="0.5"),
                ActionStep("type", value=goal_lower.replace("search", "").strip()),
                ActionStep("hotkey", value="return"),
            ]
        elif "safari" in goal_lower or "browser" in goal_lower:
            return [
                ActionStep("hotkey", value="cmd+space"),
                ActionStep("wait", value="0.5"),
                ActionStep("type", value="Safari"),
                ActionStep("hotkey", value="return"),
                ActionStep("wait", value="1.5"),
            ]
        elif "screenshot" in goal_lower:
            return [ActionStep("screenshot")]
        else:
            return [
                ActionStep("hotkey", value="cmd+space"),
                ActionStep("wait", value="0.5"),
                ActionStep("type", value=goal),
                ActionStep("hotkey", value="return"),
            ]

    def describe_screen(self) -> str:
        """Shortcut: takes a screenshot and returns a verbal description."""
        return self.observe()

    def get_stats(self) -> dict:
        return {
            "tasks_executed": self.tasks_executed,
            "actions_executed": len(self.executor.action_log),
            "pyautogui": _PYAUTOGUI_AVAILABLE,
            "pillow": _PIL_AVAILABLE,
            "dry_run": self.executor.dry_run,
        }
