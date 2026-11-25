# Данные

- `demo/` — демо-квартира 48 м² из презентации: `demo_48.json` (план) и
  `demo_48_state.json` (полное состояние: бриф, ограничения из правки, расстановка).
- `raw_plans/` — картинки планов застройщиков как есть, без разметки (вход для `aidesign recognize`).
- `plans/` — размеченные планы после `aidesign convert-labelme`, если понадобятся.
- `external/` — сырые датасеты (ResPlan.pkl, CubiCasa5K). В git не попадают.
- `resplan/` — результат `aidesign convert-resplan`. В git не попадает: генерируется заново.
