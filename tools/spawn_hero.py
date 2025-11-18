#!/usr/bin/env python3
"""
Spawn a vehicle in a running CARLA world and set its role_name to 'hero'.

This is a robust, minimal script intended for local testing with `manual_control.py`.
It prints actor lists before and after spawning so you can verify effects.
"""

import argparse
import random
import time
import sys

try:
    import carla
except Exception:
    print("Failed to import carla. Make sure PYTHONPATH includes CARLA PythonAPI/egg and that CARLA is installed.")
    raise


def parse_args():
    p = argparse.ArgumentParser(description="Spawn a hero vehicle in CARLA")
    p.add_argument('--host', default='127.0.0.1', help='CARLA host (default: 127.0.0.1)')
    p.add_argument('--port', default=2000, type=int, help='CARLA port (default: 2000)')
    p.add_argument('--filter', default='vehicle.*', help='Blueprint filter for vehicle selection')
    p.add_argument('--index', type=int, default=None, help='Spawn point index to use (default: random)')
    p.add_argument('--role-name', default='hero', help="role_name to assign to the spawned vehicle (default 'hero')")
    p.add_argument('--destroy-after', type=float, default=0.0, help='Seconds after which to destroy the spawned actor (0 = keep)')
    p.add_argument('--no-randomize-color', dest='random_color', action='store_false', help='Do not randomize color attribute')
    p.add_argument('--keep-existing', action='store_true', help='If a hero already exists, do not spawn another and print its id')
    return p.parse_args()


def list_heroes(world, role_name='hero'):
    actors = world.get_actors()
    heroes = [a for a in actors if a.attributes.get('role_name', '') == role_name]
    return heroes


def main():
    args = parse_args()

    print(f"Connecting to CARLA at {args.host}:{args.port}")
    sys.stdout.flush()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
    except Exception as e:
        print(f"Failed to get world from CARLA server at {args.host}:{args.port}: {e}")
        sys.exit(2)

    before = list_heroes(world, args.role_name)
    print(f"Heroes before spawn: {[ (a.id, a.type_id) for a in before ]}")
    sys.stdout.flush()

    if before and args.keep_existing:
        print("Hero already exists and --keep-existing specified. Exiting.")
        return

    blueprint_library = world.get_blueprint_library()
    bps = list(blueprint_library.filter(args.filter))
    if not bps:
        print(f"No blueprints found matching filter '{args.filter}'")
        sys.exit(3)

    spawn_points = []
    try:
        spawn_points = world.get_map().get_spawn_points()
    except Exception:
        spawn_points = []

    if not spawn_points:
        print("No spawn points available in the current map. Cannot spawn vehicle.")
        sys.exit(4)

    if args.index is not None:
        if args.index < 0 or args.index >= len(spawn_points):
            print(f"Index {args.index} out of range (0..{len(spawn_points)-1})")
            sys.exit(5)
        spawn_pt = spawn_points[args.index]
    else:
        spawn_pt = random.choice(spawn_points)

    print(f"Found {len(bps)} blueprints matching filter '{args.filter}'")
    print(f"Using spawn point index {spawn_points.index(spawn_pt)} at {spawn_pt}")
    sys.stdout.flush()

    attempts = []
    random.shuffle(bps)
    actor = None
    for bp in bps:
        print(f"Trying blueprint {bp.id}")
        sys.stdout.flush()
        if not bp.id.startswith('vehicle.'):
            continue

        bp_copy = bp

        # randomize color if supported
        if args.random_color:
            try:
                attr = bp_copy.get_attribute('color')
            except Exception:
                attr = None
            if attr is not None:
                try:
                    colors = list(attr.recommended_values)
                except Exception:
                    colors = None
                if colors:
                    try:
                        bp_copy.set_attribute('color', random.choice(colors))
                    except Exception:
                        pass

        # set role_name
        try:
            bp_copy.set_attribute('role_name', args.role_name)
        except Exception:
            pass

        # try safe spawn first
        try:
            actor = world.try_spawn_actor(bp_copy, spawn_pt)
            if actor is None:
                print(f"try_spawn_actor returned None for blueprint {bp.id}")
                try:
                    actor = world.spawn_actor(bp_copy, spawn_pt)
                    print(f"spawn_actor succeeded for blueprint {bp.id} -> id={actor.id}")
                except Exception as e:
                    print(f"spawn_actor also failed for blueprint {bp.id}: {e}")
                    actor = None
            else:
                print(f"try_spawn_actor succeeded for blueprint {bp.id} -> id={actor.id}")
            sys.stdout.flush()
        except Exception as e:
            print(f"try_spawn_actor raised for blueprint {bp.id}: {e}")
            sys.stdout.flush()
            actor = None

        if actor is not None:
            print(f"Spawned actor id={actor.id} type={bp_copy.id}")
            sys.stdout.flush()
            break
        else:
            attempts.append(bp.id)

    if actor is None:
        print(f"Failed to spawn actor. Tried blueprints: {attempts}")
        sys.exit(6)

    after = list_heroes(world, args.role_name)
    print(f"Heroes after spawn: {[ (a.id, a.type_id) for a in after ]}")
    sys.stdout.flush()

    try:
        if args.destroy_after > 0:
            print(f"Will destroy actor id={actor.id} after {args.destroy_after} seconds")
            sys.stdout.flush()
            time.sleep(args.destroy_after)
            try:
                actor.destroy()
                print(f"Destroyed actor id={actor.id}")
            except Exception as e:
                print(f"Failed to destroy actor id={actor.id}: {e}")
        else:
            print(f"Spawned actor id={actor.id}. Leaving it alive (destroy-after=0).")
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")


if __name__ == '__main__':
    main()
