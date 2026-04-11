# Deploy Backend en Render + Supabase

## 1) Supabase
- Crea un proyecto en Supabase.
- Copia tu cadena de conexion Postgres (`Direct connection`).
- Asegurate de usar `sslmode=require`, por ejemplo:

```env
DATABASE_URL=postgresql://postgres:TU_PASSWORD@db.TU-PROYECTO.supabase.co:5432/postgres?sslmode=require
```

## 2) Render (Web Service)
- En Render crea un servicio tipo `Web Service`.
- Conecta este repositorio.
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
uvicorn app.server:app --host 0.0.0.0 --port $PORT
```

## 3) Variables de entorno en Render
- `DATABASE_URL` = URL de Supabase con `sslmode=require`
- `MODELS_DIR` = `models_mix2` (o la carpeta real que uses)
- `POSE_WEIGHTS` = `yolo11s-pose.pt`
- `FUSION_W` = `0.50`
- `BOSS_CODE` = tu codigo para registro de jefe
- `CORS_ORIGINS` = URL de tu frontend (por ejemplo, dominio de Vercel)

## 4) Blueprint (opcional)
- Puedes usar `render.yaml` para crear el servicio con configuracion base.
- Los secretos (`DATABASE_URL`, `BOSS_CODE`) se cargan manualmente en Render.
