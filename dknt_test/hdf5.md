HDF5 Format
===

> Refer to [intro](https://support.hdfgroup.org/documentation/hdf5/latest/_intro_h_d_f5.html)

HDF5 stands for High-performance data management and storage suite v.5. It features heterogeneous data, cross platform, fast I/O and is designed for managing bit data.

One can use [HDFView](https://support.hdfgroup.org/releases/hdfview/v3_3/v3_3_2/downloads/index.html) to browse a HDF5 file.

> The installed `HDFView` can be found in application but can not be directly executed in terminal. Don't use the `HDFView` downloaded from `apt` since it seems to be broken. Moreover, the image list in the rdt dataset can not be displayed inside  `HDFView` as images are encoded.

An HDF5 file (an object in itself) can be thought of as a container (or group) that holds a variety of heterogeneous data objects (or datasets). The datasets can be images, tables, graphs, and even documents.

Two primary objects in the HDF5 Data Model: Group, Dataset. Group is like directory in a UNIX system. Dataset consists of raw Data and Metadata. Metadata: Dataspace, Datatype, Properties and Attrributes (optional).





