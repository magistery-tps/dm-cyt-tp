# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - DAtamining en Ciencia y Teconologia

## Trabajos Prácticos

* TP1: Microestados de EEG
* TP2: Pte.

## Pre-Requisitos

* git
* conda

## Comenzando

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
$ conda activate dm-cyt-tp
```

**Paso 4**: Sobre el directorio del proyecto levantamos jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Paso 5**: Ir a http://localhost:8888.... como se indica en la consola.

## Tema Material Darker para Jupyter Lab

**Paso 1**: Instalar tema.

```bash
$ jupyter labextension install @oriolmirosa/jupyterlab_materialdarker
```

**Paso 2**: Reiniciar Jupyter Lab

