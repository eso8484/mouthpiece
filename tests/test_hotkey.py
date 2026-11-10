import pytest

from mouthpiece.hotkey import Action, Hotkey, HotkeyError, HotkeyStateMachine
from mouthpiece.keycodes import CODES as C

RCTRL, LCTRL, LSHIFT, SPACE = C["KEY_RIGHTCTRL"], C["KEY_LEFTCTRL"], C["KEY_LEFTSHIFT"], C["KEY_SPACE"]
KEY_C, ESC, F7, LMETA, LALT = C["KEY_C"], C["KEY_ESC"], C["KEY_F7"], C["KEY_LEFTMETA"], C["KEY_LEFTALT"]


def actions(m, *events):
    out = []
    for code, value in events:
        out += [a for a, _ in m.feed(code, value)]
    return out


def test_parse_variants():
    assert Hotkey.parse("RIGHTCTRL").codes == {RCTRL}
    assert Hotkey.parse("rctrl").codes == {RCTRL}
    assert Hotkey.parse("Right Ctrl").codes == {RCTRL}
    hk = Hotkey.parse("ctrl+space")
    assert hk.groups == (frozenset({LCTRL, RCTRL}), frozenset({SPACE}))
    assert Hotkey.parse("SHIFT+F7").pretty() == "Shift + F7"
    assert Hotkey.parse("super+alt").modifier_only
    assert Hotkey.parse("MOUSE_SIDE").uses_mouse
    assert Hotkey.parse("KEY_F9").codes == {C["KEY_F9"]}


def test_parse_errors():
    with pytest.raises(HotkeyError):
        Hotkey.parse("")
    with pytest.raises(HotkeyError):
        Hotkey.parse("CTRL+NOPE")
    with pytest.raises(HotkeyError):
        Hotkey.parse("CTRL+CTRL")


def test_from_codes_keeps_sides_and_orders_modifiers():
    hk = Hotkey.from_codes([C["KEY_F7"], LSHIFT])
    assert hk.spec == "LEFTSHIFT+F7"


def test_hold_press_release():
    m = HotkeyStateMachine(Hotkey.parse("RIGHTCTRL"))
    assert actions(m, (RCTRL, 1)) == [Action.PRESS]
    assert actions(m, (RCTRL, 2), (RCTRL, 2)) == []  # autorepeat ignored
    assert actions(m, (RCTRL, 0)) == [Action.RELEASE]


def test_other_key_while_held_is_interrupt():
    m = HotkeyStateMachine(Hotkey.parse("RIGHTCTRL"))
    assert actions(m, (RCTRL, 1), (KEY_C, 1), (KEY_C, 0), (RCTRL, 0)) == [Action.PRESS, Action.INTERRUPT, Action.RELEASE]


def test_extra_modifier_held_first_prevents_trigger():
    m = HotkeyStateMachine(Hotkey.parse("RIGHTCTRL"))
    assert actions(m, (LSHIFT, 1), (RCTRL, 1), (RCTRL, 0), (LSHIFT, 0)) == []


def test_combo_any_order():
    m = HotkeyStateMachine(Hotkey.parse("CTRL+SUPER"))
    assert actions(m, (LMETA, 1), (LCTRL, 1)) == [Action.PRESS]
    assert actions(m, (LMETA, 0)) == [Action.RELEASE]
    assert actions(m, (LCTRL, 0)) == []


def test_generic_modifier_either_side():
    m = HotkeyStateMachine(Hotkey.parse("CTRL+SPACE"))
    assert actions(m, (RCTRL, 1), (SPACE, 1)) == [Action.PRESS]
    assert actions(m, (SPACE, 0)) == [Action.RELEASE]


def test_same_key_on_two_keyboards():
    m = HotkeyStateMachine(Hotkey.parse("RIGHTCTRL"))
    assert actions(m, (RCTRL, 1), (RCTRL, 1)) == [Action.PRESS]
    assert actions(m, (RCTRL, 0)) == []          # still held on the other keyboard
    assert actions(m, (RCTRL, 0)) == [Action.RELEASE]


def test_cancel_key():
    m = HotkeyStateMachine(Hotkey.parse("RIGHTCTRL"), frozenset({ESC}))
    assert actions(m, (RCTRL, 1), (ESC, 1)) == [Action.PRESS, Action.CANCEL, Action.INTERRUPT]
    assert actions(m, (ESC, 0), (RCTRL, 0)) == [Action.RELEASE]


def test_device_unplug_releases():
    m = HotkeyStateMachine(Hotkey.parse("RIGHTCTRL"))
    actions(m, (RCTRL, 1))
    assert [a for a, _ in m.drop_codes({RCTRL})] == [Action.RELEASE]


def test_mouse_clicks_ignored_but_hotkey_button_tracked():
    m = HotkeyStateMachine(Hotkey.parse("MOUSE_SIDE"))
    assert actions(m, (C["BTN_LEFT"], 1), (C["BTN_LEFT"], 0)) == []
    assert actions(m, (C["BTN_SIDE"], 1), (C["BTN_SIDE"], 0)) == [Action.PRESS, Action.RELEASE]
