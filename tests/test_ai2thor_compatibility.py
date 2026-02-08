"""
Test suite for AI2-THOR 5.0.0 compatibility.
Following TDD methodology - these tests define the expected behavior.
"""
import pytest
import sys
import platform

sys.path.insert(0, '.')
sys.path.insert(0, './alfred')


class TestTeleportActionConversion:
    """Test TeleportFull action format conversion from old to new API."""

    def test_convert_scalar_rotation_to_vector3(self):
        """Rotation should be converted from scalar to Vector3 dict."""
        from alfred.env.thor_env import ThorEnv

        # Create a minimal ThorEnv instance just for the conversion method
        # We'll test the conversion logic without starting the simulator
        old_action = {
            'action': 'TeleportFull',
            'x': 1.0,
            'y': 0.9,
            'z': -1.5,
            'rotation': 90,  # Old format: scalar
            'horizon': 30,
        }

        # The _convert_teleport_action should handle this
        env = object.__new__(ThorEnv)
        env._convert_teleport_action = ThorEnv._convert_teleport_action.__get__(env, ThorEnv)

        new_action = env._convert_teleport_action(old_action)

        # Verify rotation is now Vector3
        assert isinstance(new_action['rotation'], dict)
        assert new_action['rotation'] == {'x': 0, 'y': 90, 'z': 0}

    def test_remove_rotateOnTeleport_parameter(self):
        """rotateOnTeleport parameter should be removed."""
        from alfred.env.thor_env import ThorEnv

        old_action = {
            'action': 'TeleportFull',
            'x': 1.0,
            'y': 0.9,
            'z': -1.5,
            'rotation': 180,
            'horizon': 30,
            'rotateOnTeleport': True,  # This should be removed
        }

        env = object.__new__(ThorEnv)
        env._convert_teleport_action = ThorEnv._convert_teleport_action.__get__(env, ThorEnv)

        new_action = env._convert_teleport_action(old_action)

        # rotateOnTeleport should not be in the new action
        assert 'rotateOnTeleport' not in new_action

    def test_add_standing_parameter(self):
        """standing=True should be added to TeleportFull actions."""
        from alfred.env.thor_env import ThorEnv

        old_action = {
            'action': 'TeleportFull',
            'x': 1.0,
            'y': 0.9,
            'z': -1.5,
            'rotation': 270,
            'horizon': 0,
        }

        env = object.__new__(ThorEnv)
        env._convert_teleport_action = ThorEnv._convert_teleport_action.__get__(env, ThorEnv)

        new_action = env._convert_teleport_action(old_action)

        # standing should be True
        assert 'standing' in new_action
        assert new_action['standing'] is True

    def test_preserve_position_coordinates(self):
        """Position coordinates (x, y, z) should be preserved."""
        from alfred.env.thor_env import ThorEnv

        old_action = {
            'action': 'TeleportFull',
            'x': 2.5,
            'y': 1.2,
            'z': -3.0,
            'rotation': 0,
            'horizon': 45,
        }

        env = object.__new__(ThorEnv)
        env._convert_teleport_action = ThorEnv._convert_teleport_action.__get__(env, ThorEnv)

        new_action = env._convert_teleport_action(old_action)

        assert new_action['x'] == 2.5
        assert new_action['y'] == 1.2
        assert new_action['z'] == -3.0
        assert new_action['horizon'] == 45

    def test_handle_already_vector3_rotation(self):
        """If rotation is already Vector3, it should be preserved."""
        from alfred.env.thor_env import ThorEnv

        old_action = {
            'action': 'TeleportFull',
            'x': 1.0,
            'y': 0.9,
            'z': -1.5,
            'rotation': {'x': 0, 'y': 90, 'z': 0},  # Already Vector3
            'horizon': 30,
            'standing': True,
        }

        env = object.__new__(ThorEnv)
        env._convert_teleport_action = ThorEnv._convert_teleport_action.__get__(env, ThorEnv)

        new_action = env._convert_teleport_action(old_action)

        assert new_action['rotation'] == {'x': 0, 'y': 90, 'z': 0}


class TestObjectStateHelpers:
    """Test helper functions that replace SetStateOfAllObjects."""

    def test_set_objects_dirty_helper_exists(self):
        """_set_objects_dirty helper should exist."""
        from alfred.env.thor_env import ThorEnv
        assert hasattr(ThorEnv, '_set_objects_dirty')

    def test_empty_fillable_objects_helper_exists(self):
        """_empty_fillable_objects helper should exist."""
        from alfred.env.thor_env import ThorEnv
        assert hasattr(ThorEnv, '_empty_fillable_objects')


class TestControllerInitialization:
    """Test AI2-THOR Controller initialization for different platforms."""

    def test_controller_init_params_no_xdisplay_on_macos(self):
        """On macOS, x_display should not be passed to Controller."""
        if platform.system() != 'Darwin':
            pytest.skip("This test is for macOS only")

        # Verify our ThorEnv doesn't pass x_display on macOS
        import inspect
        from alfred.env.thor_env import ThorEnv

        source = inspect.getsource(ThorEnv.__init__)
        # Should check platform before adding x_display
        assert 'platform.system()' in source or "platform.system() == 'Linux'" in source

    def test_ai2thor_version_is_5(self):
        """Verify AI2-THOR 5.0.0 is installed."""
        import ai2thor
        version = ai2thor.__version__
        assert version.startswith('5.'), f"Expected AI2-THOR 5.x, got {version}"


class TestVisibilityDistanceParameter:
    """Test that visibilityDistance parameter is used correctly (camelCase)."""

    def test_reset_uses_camelcase_visibility_distance(self):
        """reset() should use visibilityDistance not visibility_distance."""
        import inspect
        from alfred.env.thor_env import ThorEnv

        source = inspect.getsource(ThorEnv.reset)
        # Should use camelCase
        assert 'visibilityDistance' in source
        # Should NOT use snake_case in the actual step call
        # (It's OK to have it as parameter name, but not in the dict)


class TestGameStateBaseCompatibility:
    """Test game_state_base.py compatibility with AI2-THOR 5.x."""

    def test_no_set_state_of_all_objects(self):
        """SetStateOfAllObjects should not be used (it's removed in 5.x)."""
        with open('alfred/gen/game_states/game_state_base.py', 'r') as f:
            content = f.read()

        # SetStateOfAllObjects should be replaced
        assert 'SetStateOfAllObjects' not in content, \
            "SetStateOfAllObjects is removed in AI2-THOR 5.x, use individual object actions"

    def test_teleport_uses_vector3_rotation(self):
        """TeleportFull in game_state_base should use Vector3 rotation."""
        with open('alfred/gen/game_states/game_state_base.py', 'r') as f:
            content = f.read()

        # Should not have rotateOnTeleport
        assert 'rotateOnTeleport' not in content, \
            "rotateOnTeleport is removed in AI2-THOR 5.x"


class TestRequirementsFile:
    """Test that requirements specify AI2-THOR 5.x."""

    def test_main_requirements_ai2thor_version(self):
        """Main requirements.txt should specify ai2thor>=5.0.0."""
        with open('requirements.txt', 'r') as f:
            content = f.read()

        # Should specify 5.x
        assert 'ai2thor' in content
        assert '5.0.0' in content or 'ai2thor>=5' in content

    def test_alfred_requirements_ai2thor_version(self):
        """alfred/requirements.txt should specify ai2thor>=5.0.0."""
        with open('alfred/requirements.txt', 'r') as f:
            content = f.read()

        # Should specify 5.x
        assert 'ai2thor' in content
        assert '5.0.0' in content or 'ai2thor>=5' in content


# Integration tests (require actual AI2-THOR simulator)
@pytest.mark.integration
class TestAI2THORIntegration:
    """Integration tests that require the actual AI2-THOR simulator."""

    @pytest.fixture
    def controller(self):
        """Create a basic AI2-THOR controller for testing."""
        from ai2thor.controller import Controller
        c = Controller(
            quality='Very Low',
            width=300,
            height=300,
        )
        yield c
        c.stop()

    def test_teleport_full_with_vector3_rotation(self, controller):
        """Test TeleportFull works with Vector3 rotation format."""
        controller.reset("FloorPlan1")

        event = controller.step(
            action="TeleportFull",
            x=1.0,
            y=0.9,
            z=-1.5,
            rotation={'x': 0, 'y': 90, 'z': 0},
            horizon=30,
            standing=True
        )

        assert event.metadata['lastActionSuccess'], \
            f"TeleportFull failed: {event.metadata.get('errorMessage', 'Unknown error')}"

    def test_dirty_object_action(self, controller):
        """Test DirtyObject action works (replacement for SetStateOfAllObjects)."""
        controller.reset("FloorPlan1")

        # Find a dirtyable object
        dirtyable_obj = None
        for obj in controller.last_event.metadata['objects']:
            if obj.get('dirtyable', False):
                dirtyable_obj = obj
                break

        if dirtyable_obj is None:
            pytest.skip("No dirtyable objects in scene")

        event = controller.step(
            action="DirtyObject",
            objectId=dirtyable_obj['objectId'],
            forceAction=True
        )

        # DirtyObject should succeed
        assert event.metadata['lastActionSuccess'], \
            f"DirtyObject failed: {event.metadata.get('errorMessage', 'Unknown error')}"

    def test_scene_reset_and_navigation(self, controller):
        """Test basic scene reset and navigation work."""
        controller.reset("FloorPlan1")

        # Get reachable positions
        event = controller.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]

        assert len(positions) > 0, "No reachable positions found"

        # Try to teleport to first reachable position
        pos = positions[0]
        event = controller.step(
            action="TeleportFull",
            x=pos['x'],
            y=pos['y'],
            z=pos['z'],
            rotation={'x': 0, 'y': 0, 'z': 0},
            horizon=0,
            standing=True
        )

        assert event.metadata['lastActionSuccess'], \
            f"Teleport to reachable position failed: {event.metadata.get('errorMessage', 'Unknown error')}"


if __name__ == '__main__':
    # Run unit tests only (not integration tests)
    pytest.main([__file__, '-v', '-m', 'not integration'])
