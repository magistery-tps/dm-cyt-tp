# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Datamining en Ciencia y Tecnologia

[Wiki](https://github.com/magistery-tps/dm-cyt-tp/wiki)

## Trabajos Prácticos

* **TP1**: Microestados de EEG.
    * [Consigna](https://github.com/magistery-tps/dm-cyt-tp/blob/main/docs/DMCT_preTP1_2021.pdf)
    * [Informe]() 

## Pre-Requisitos

* [git](https://git-scm.com/downloads)
* [anaconda](https://www.anaconda.com/products/individual)/[minconda](https://docs.conda.io/en/latest/miniconda.html)

## Comenzando

### Video

[Paso a Paso (Windows)](https://www.youtube.com/watch?v=O8YXuHNdIIk)

### Pasos

**Paso 1**: Descargar el repositorio.

```bash
$ git clone https://github.com/magistery-tps/dm-cyt-tp.git
$ cd dm-cyt-tp
```

**Paso 2**: Crear environment de dependencias para el proyecto (Parado en el directorio del proyecto).

```bash
$ conda env create -f environment.yml
```

**Paso 3**: Activamos el entorno donde se encuentran instaladas las dependencias del proyecto.

```bash
$ conda activate dm-cyt-tp1
```

**Paso 4**: Descomprimir el dataset:

```bash
$ unzip datasets/procesados-20210908T224817Z-001.zip
$ mv procesados dataset
```
_Nota_: **En windows se puede hacer desde el explorador de archivos**.

**Paso 5**: Sobre el directorio del proyecto levantamos jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Paso 6**: Ir a http://localhost:8888.... como se indica en la consola.

## Tema Material Darker para Jupyter Lab

**Paso 1**: Instalar tema.

```bash
$ jupyter labextension install @oriolmirosa/jupyterlab_materialdarker
```

**Paso 2**: Reiniciar Jupyter Lab

