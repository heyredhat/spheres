cd docs
make html
cd ..

python setup.py sdist bdist_wheels
twine upload -r testpypi dist/*