set -e

echo "Восстановление Poetry-окружения"
cd poetry_model
poetry env use python3.9
poetry install
poetry run python linear_model_poetry.py

cd ..

echo "Восстановление Conda-окружения"
conda env create -f conda_model/environment.yml || \
conda env update -f conda_model/environment.yml --prune

conda run -n conda-model \
    python conda_model/linear_model_conda.py

echo "Проверка завершена успешно"
