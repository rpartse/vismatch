vismatch
========

Vis(ion)Match(ers) is a unified API for 50+ image matching models with a consistent interface.

.. code-block:: python

   from vismatch import get_matcher

   matcher = get_matcher("superpoint-lightglue", device="cuda")
   img0 = matcher.load_image("img0.jpg", resize=512)
   img1 = matcher.load_image("img1.jpg", resize=512)
   result = matcher(img0, img1)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   model_details

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/vismatch
   api/vismatch.base_matcher
   api/vismatch.utils
   api/vismatch.viz
   api/vismatch.im_models

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
