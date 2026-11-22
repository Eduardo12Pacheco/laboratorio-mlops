# Política de seguridad

## Versiones soportadas

Se soportan el último release y la rama `main` actual. Es un laboratorio educativo local-first, no un
servicio alojado.

## Reportar una vulnerabilidad

Abre un aviso de seguridad privado en GitHub (Security → Advisories) o contacta al mantenedor a través de
su perfil de GitHub. Por favor, no abras un issue público con los detalles de un exploit antes de que
haya una corrección disponible. La respuesta inicial se espera en pocos días.

## Modelo de confianza

Es una demo local-first. Los artefactos de modelo almacenados en MLflow se tratan como estado local de
confianza: cargar un modelo ejecuta código basado en pickle. Ejecuta el stack solo contra registros y
artefactos que hayas creado o en los que confíes, y nunca apuntes `MLL_MLFLOW_TRACKING_URI` a un servidor
no confiable. La variable opcional `MLL_ADMIN_API_KEY` solo protege `POST /v1/admin/reload`; no es un
sistema de autenticación para el resto de la API, que por defecto escucha solo en loopback.

## Secretos y cuentas

- Cero claves o cuentas obligatorias. Todo se ejecuta sin conexión, con el dataset incluido y con pruebas
  deterministas.
- `.env.example` es la única plantilla de entorno que se versiona; `.env` está en `.gitignore`. Nunca
  comprometas un `MLL_ADMIN_API_KEY` real ni una credencial de registro.
- La API expone `GET /metrics` sin autenticación, como el resto de los endpoints locales; no lo publiques
  en una interfaz pública.
- La telemetría está desactivada por defecto para MLflow, Evidently/iterative-telemetry y Streamlit, para
  que el laboratorio no envíe datos a ningún servicio externo.

## Documentación y assets de reportes

Los archivos JSON/Markdown comprometidos en `docs/results/` se generan a partir del dataset incluido de
confianza y del drill local. Los reportes HTML curados de Evidently en `docs/assets/reports/` no contienen
scripts remotos: su bundle de JavaScript se extrae a un archivo local para que la documentación funcione
sin conexión. Revisa el HTML generado antes de publicar un release; es salida estática, nunca código
ejecutable del lado del servidor. La atribución y la licencia de los datos se registran en `docs/DATA.md`.

## Cadena de suministro y postura de contenedores

- Las dependencias de runtime y de desarrollo están fijadas en `uv.lock`; CI ejecuta `uv lock --check`,
  `pip-audit` sobre los requisitos de runtime exportados y un escaneo gitleaks del historial.
- La imagen de contenedor se ejecuta con el usuario no root `mll` (uid 10001). Cada servicio de Compose
  elimina todas las capacidades, activa `no-new-privileges`, usa un sistema de archivos raíz de solo
  lectura con un tmpfs en `/tmp` y publica puertos solo en `127.0.0.1`.
- El estado operativo de SQLite y los artefactos de MLflow viven en el volumen con nombre `mll-data`; son
  estado local de confianza y el laboratorio no los respalda.
