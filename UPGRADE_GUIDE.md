# NumPy 2.x Upgrade Guide

## What Changed

The project has been upgraded to use NumPy 2.x with modern TensorFlow 2.17+ to resolve compatibility issues.

### Updated Dependencies

| Package | Old Version | New Version |
|---------|-------------|-------------|
| TensorFlow (CPU) | 2.12.0 | 2.17.0+ |
| Keras | 2.12.0 | 3.0.0+ |
| NumPy | 1.23.5 | 2.0.0+ |
| scikit-learn | 1.2.2 | 1.5.0+ |
| scikit-image | 0.19.3 | 0.24.0+ |
| pandas | 1.4.0+ | 2.0.0+ |

## Upgrading Your Local Environment

### Windows

Run the upgrade script:
```bash
upgrade_env.bat
```

### Linux/Mac

Run the upgrade script:
```bash
chmod +x upgrade_env.sh
./upgrade_env.sh
```

### Manual Steps

If the scripts don't work, follow these manual steps:

1. **Activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Uninstall old packages:**
   ```bash
   pip uninstall -y tensorflow tensorflow-cpu keras numpy scikit-learn scikit-image
   ```

4. **Install new requirements:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Test compatibility:**
   ```bash
   python test_model_compatibility.py
   ```

## Retraining Models

If the compatibility test shows models failed to load, retrain them:

```bash
# Retrain all models
python train_disease_models.py --models all

# Or retrain specific models
python train_disease_models.py --models xception,resnet50,mobilenet,cnn,svm,rf
```

## Deployment

The changes are already deployed to Streamlit Cloud. The app will automatically:
- Install the updated dependencies from `environment.yml`
- Use models that are compatible with NumPy 2.x

## Troubleshooting

### Models Still Fail to Load

If models still fail after upgrade:
1. Make sure you're using the virtual environment
2. Check Python version (should be 3.10)
3. Retrain all models with `python train_disease_models.py --models all`

### TensorFlow Installation Issues

If TensorFlow fails to install:
- Make sure you're using Python 3.10 (not 3.13)
- Try: `pip install tensorflow-cpu==2.17.0` specifically

### Import Errors

If you see import errors:
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

## What Was Fixed

**Problem:** Models trained with NumPy 2.x couldn't load on Streamlit Cloud with NumPy 1.x.

**Error:** `No module named 'numpy._core'`

**Solution:** Upgraded entire stack to NumPy 2.x compatible versions:
- ✅ Streamlit Cloud now uses NumPy 2.x
- ✅ Local environment upgraded to match
- ✅ Models trained locally work on deployment
- ✅ Deprecation warnings fixed (use_container_width → width)

## Resources

- [TensorFlow 2.17 Release Notes](https://github.com/tensorflow/tensorflow/releases/tag/v2.17.0)
- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [Keras 3 Documentation](https://keras.io/keras_3/)
