extends SceneTree
## Regression test: the RL tab must not render blank.
##
## rl_trainer.gd's root is a plain Control (not a Container), so its
## ScrollContainer child only gets a rect if it is anchored to the full rect.
## This test loads the real main-menu scene at the project window size,
## switches to the RL tab, and asserts the page, its ScrollContainer and the
## content column all lay out (the pre-fix failure mode was page=1520x796
## but scroll=772x0 -> every card clipped away).
##
## Run:  godot --headless --script res://tests/validate_rl_layout.gd

const WINDOW_SIZE := Vector2i(1600, 900)
const PIN_FRAMES_MAX := 60
const SETTLE_FRAMES := 4
const VERIFY_AFTER_SWITCH := 8

var _frames := 0
var _pinned := false
var _settle := 0


func _initialize() -> void:
	var err := change_scene_to_file("res://scenes/main_menu.tscn")
	if err != OK:
		print("LAYOUT: FAIL — could not load main menu (err %d)" % err)
		quit(1)


func _process(_delta: float) -> bool:
	_frames += 1
	if current_scene == null:
		return false
	if not _pinned:
		# Headless windows collapse to the content minimum size; pin the real
		# project resolution so page-holder expansion behaves like the GUI.
		root.size = WINDOW_SIZE
		if root.size == WINDOW_SIZE:
			_pinned = true
			print("LAYOUT: window pinned at frame %d" % _frames)
		elif _frames > PIN_FRAMES_MAX:
			print("LAYOUT: FAIL — could not pin window (root.size=%s)" % root.size)
			quit(1)
			return true
		return false
	_settle += 1
	if _settle == SETTLE_FRAMES:
		current_scene.call("_switch_page", "RL")
	elif _settle == VERIFY_AFTER_SWITCH:
		return _verify()
	return false


func _verify() -> bool:
	var menu := current_scene
	var page: Control = menu.get("_pages")["RL"]
	var scroll := page.get_child(0) as Control
	var vbox := scroll.get_child(0) as Control
	var card: Control = null
	for child in vbox.get_children():
		if child is PanelContainer:
			card = child
			break
	print("LAYOUT: page=%s scroll=%s vbox=%s card=%s" % [
		page.size, scroll.size, vbox.size, card.size if card != null else "null"])

	var failures: Array[String] = []
	if not page.visible:
		failures.append("RL page is not visible after switching tabs")
	if page.size.y <= 0.0:
		failures.append("RL page has no height (page=%s)" % page.size)
	if absf(scroll.size.y - page.size.y) >= 1.0:
		failures.append("ScrollContainer does not fill the page vertically (scroll=%s page=%s)" % [scroll.size, page.size])
	if absf(scroll.size.x - page.size.x) >= 1.0:
		failures.append("ScrollContainer does not fill the page horizontally (scroll=%s page=%s)" % [scroll.size, page.size])
	if vbox.size.y <= scroll.size.y:
		failures.append("content column (%s) does not exceed the viewport — cards clipped" % vbox.size)
	if card == null or card.size.y <= 0.0:
		failures.append("hyper-parameter card did not lay out")

	if failures.is_empty():
		print("LAYOUT: PASS — RL tab fills the page")
		quit(0)
		return true
	for f in failures:
		print("LAYOUT: FAIL — ", f)
	quit(1)
	return true
