{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
  :members:
  :private-members:
  :undoc-members:
  :show-inheritance:
  :inherited-members:



  {% block attributes %}
  {% if attributes %}
  .. rubric:: {{ _('Attributes') }}

  .. autosummary::
  {% for item in attributes %}
      ~{{ name }}.{{ item }}
  {%- endfor %}
  {% endif %}
  {% endblock %}
