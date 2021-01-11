{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   .. rubric:: Methods
   .. autosummary::

   {% for item in methods %}
   {%- if item not in ["__init__"] and item[0] != '_'%}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}

.. include:: {{fullname}}.examples
