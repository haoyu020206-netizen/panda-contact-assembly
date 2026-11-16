# Third-party notices

## MuJoCo Menagerie — Franka Emika Panda

The high-fidelity Panda model and mesh assets under
`src/panda_assembly/assets/menagerie/franka_emika_panda/` are copied from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
at commit `370c43965323c0944b433b202aac5058b5f907a9`.

- Upstream model: `franka_emika_panda/panda_nohand.xml`
- Upstream license: Apache License 2.0
- Local derived scene: `src/panda_assembly/assets/panda_menagerie.xml`
- Local changes: assembly table and socket, rigid peg tool, named control sites, torque
  actuators, camera and rendering settings

The upstream license text is preserved at
`src/panda_assembly/assets/menagerie/franka_emika_panda/LICENSE`.
