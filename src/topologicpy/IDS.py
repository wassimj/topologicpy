# Copyright (C) 2026
# Wassim Jabi <wassim.jabi@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free Software
# Foundation, either version 3.0 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""Independent IDS 1.0 assessment for TopologicPy IFC models.

IFC parsing is delegated to topologicpy.IFC.IFC. No IFC SDK or third-party
IDS evaluator is imported. Helpers stay with this public class/module.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as SafeET
from elementpath.regex import translate_pattern
import xmlschema
from topologicpy.IFC import IFC, IFCLogicalUnknown




class IDS:
    """Read IDS requirements and assess a TopologicPy IFCModel.

    Public methods return ordinary dictionaries suitable for application APIs,
    reports and other TopologicPy integrations. They do not require geometry.
    """
    VERSION='TopologicPy IDS 1.0.0'

    @dataclass
    class _Value:
        value: object
        datatype: str = ''
        unit: object = None
        supported: bool = True
        alternatives: object = None

    NS='{http://standards.buildingsmart.org/IDS}'
    XS='{http://www.w3.org/2001/XMLSchema}'

    @staticmethod
    @lru_cache(maxsize=3)
    def _schema(name):
        canonical='IFC4X3_ADD2' if name=='IFC4X3' else name
        path=Path(__file__).with_name('ids_schemas')/(canonical+'.json')
        if not path.is_file():raise ValueError('The IFC schema is not supported: '+str(name))
        return json.loads(path.read_text(encoding='utf-8'))

    @staticmethod
    @lru_cache(maxsize=1)
    def _xsd():
        return xmlschema.XMLSchema(str(Path(__file__).with_name('ids_schemas')/'ids.xsd'),allow='local')

    @staticmethod
    @lru_cache(maxsize=1)
    def _ifc2x3_mapping():
        return json.loads(Path(__file__).with_name('ids_schemas').joinpath('ifc2x3-mapping.json').read_text())

    @staticmethod
    def ByPath(path):
        """Read and schema-validate an IDS 1.0 document without network access."""
        try:
            root=SafeET.parse(path).getroot()
            if root.tag!=IDS.NS+'ids':raise ValueError('Wrong namespace')
            IDS._xsd().validate(str(path))
            return root
        except Exception as error:
            raise ValueError('The IDS is invalid or unsupported. Supply an IDS 1.0 file that passes XML schema validation.') from error

    @staticmethod
    def _constraint(node):
        if node is None:return None
        simple=node.find(IDS.NS+'simpleValue')
        if simple is not None:return {'simple':simple.text or ''}
        restriction=node.find(IDS.XS+'restriction')
        if restriction is None:return None
        result={'base':restriction.get('base','xs:string').split(':')[-1],'rules':defaultdict(list)}
        for child in restriction:result['rules'][child.tag.split('}')[-1]].append(child.get('value',''))
        return result

    @staticmethod
    def _equivalent(value,target):
        if not math.isfinite(value) or not math.isfinite(target):return value==target and not math.isnan(target)
        radius=abs(target)*1e-6
        return math.nextafter(target-radius-1e-6,-math.inf)<=value<=math.nextafter(target+radius+1e-6,math.inf)

    @staticmethod
    def _cast(text,value):
        if type(value) is bool:return {'true':True,'false':False,'1':True,'0':False}.get(str(text),object())
        if type(value) is int:
            return int(text) if re.fullmatch(r'[+-]?[0-9]+',str(text)) else object()
        if isinstance(value,float):
            if not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|NaN|[+-]INF',str(text)):return object()
            try:return float(text)
            except ValueError:return object()
        return str(text)

    @staticmethod
    def _match(constraint,value):
        if constraint is None:return True
        if value is None:return False
        if isinstance(value,list):return any(IDS._match(constraint,v) for v in value)
        if isinstance(value,tuple):return False
        if 'simple' in constraint:
            target=IDS._cast(constraint['simple'],value)
            return IDS._equivalent(value,target) if isinstance(value,float) and isinstance(target,float) else value==target
        for kind,options in constraint['rules'].items():
            if kind=='enumeration':
                if not any(IDS._match({'simple':option},value) for option in options):return False
            elif kind=='pattern':
                if not isinstance(value,str) or not any(re.fullmatch(translate_pattern(option,back_references=False,lazy_quantifiers=False,anchors=False),value) for option in options):return False
            elif kind in {'length','minLength','maxLength'}:
                n=len(str(value));limit=int(options[0])
                if kind=='length' and n!=limit or kind=='minLength' and n<limit or kind=='maxLength' and n>limit:return False
            elif kind in {'minInclusive','minExclusive','maxInclusive','maxExclusive'}:
                try:n=float(value);limit=float(options[0])
                except (ValueError,TypeError):return False
                if kind=='minInclusive' and n<limit or kind=='minExclusive' and n<=limit or kind=='maxInclusive' and n>limit or kind=='maxExclusive' and n>=limit:return False
        return True

    @staticmethod
    def _primitive(schema,kind):
        kind=kind.replace('OPTIONAL ','').strip()
        seen=set()
        while kind.upper() in schema['types'] and kind.upper() not in seen:
            seen.add(kind.upper());kind=schema['types'][kind.upper()].strip()
        if re.search(r'\b(?:LIST|SET|ARRAY|BAG)\b',kind,re.I):return 'aggregate'
        if 'SELECT' in kind.upper():return 'select'
        return kind.upper()

    @staticmethod
    def _isa(ctx,entity,name):
        current=entity.type if entity else None
        while current:
            if current.upper()==name.upper():return True
            current=ctx['schema']['entities'].get(current.upper(),{}).get('parent')
        return False

    @staticmethod
    def _attr(ctx,entity,name,default=None):
        if entity is None:return default
        fields=ctx['schema']['entities'].get(entity.type,{}).get('attributes',[])
        for index,(key,kind,derived) in enumerate(fields):
            if key==name:return entity.args[index] if index<len(entity.args) and not derived else default
        return default

    @staticmethod
    def _refs(ctx,value):
        if isinstance(value,tuple) and len(value)==2 and value[0]=='REF':
            target=ctx['entities'].get(value[1]);return [target] if target else []
        if isinstance(value,(list,tuple)):
            return [e for item in value for e in IDS._refs(ctx,item)]
        return []

    @staticmethod
    def _ref(ctx,entity,name):
        values=IDS._refs(ctx,IDS._attr(ctx,entity,name));return values[0] if values else None

    @staticmethod
    def _unwrap(value):
        if isinstance(value,tuple) and value[0]=='CALL':
            args=value[2];return IDS._unwrap(args[0]) if len(args)==1 else [IDS._unwrap(v) for v in args]
        if isinstance(value,list):return [IDS._unwrap(v) for v in value]
        return value

    @staticmethod
    def _typed(value,unit=None):
        datatype=value[1] if isinstance(value,tuple) and value[0]=='CALL' else ''
        return IDS._Value(IDS._unwrap(value),datatype,unit)

    @staticmethod
    def _context(model):
        ctx={'model':model,'schema':IDS._schema(model.schema),'entities':model.entities,
             'types':{},'psets':defaultdict(list),'classifications':defaultdict(list),'materials':defaultdict(list),'parents':defaultdict(list),'units':{},'property_cache':{},'classification_cache':{}}
        for e in model.entities.values():
            attr=lambda name:IDS._attr(ctx,e,name)
            refs=lambda name:IDS._refs(ctx,attr(name))
            if e.type=='IFCRELDEFINESBYTYPE':
                targets=refs('RelatingType')
                if targets:
                    for obj in refs('RelatedObjects'):ctx['types'][obj.id]=targets[0]
            elif e.type=='IFCRELDEFINESBYPROPERTIES':
                for obj in refs('RelatedObjects'):ctx['psets'][obj.id].extend(refs('RelatingPropertyDefinition'))
            elif e.type=='IFCRELASSOCIATESCLASSIFICATION':
                for obj in refs('RelatedObjects'):ctx['classifications'][obj.id].extend(refs('RelatingClassification'))
            elif e.type=='IFCEXTERNALREFERENCERELATIONSHIP':
                for obj in refs('RelatedResourceObjects'):ctx['classifications'][obj.id].extend(refs('RelatingReference'))
            elif e.type=='IFCRELASSOCIATESMATERIAL':
                for obj in refs('RelatedObjects'):ctx['materials'][obj.id].extend(refs('RelatingMaterial'))
            elif IDS._isa(ctx,e,'IfcMaterialProperties') or IDS._isa(ctx,e,'IfcProfileProperties'):
                for obj in refs('Material')+refs('ProfileDefinition'):ctx['psets'][obj.id].append(e)
            if IDS._isa(ctx,e,'IfcTypeObject'):ctx['psets'][e.id].extend(refs('HasPropertySets'))
            relation_fields={'IFCRELAGGREGATES':('RelatingObject','RelatedObjects'),'IFCRELNESTS':('RelatingObject','RelatedObjects'),
                'IFCRELCONTAINEDINSPATIALSTRUCTURE':('RelatingStructure','RelatedElements'),'IFCRELASSIGNSTOGROUP':('RelatingGroup','RelatedObjects'),
                'IFCRELVOIDSELEMENT':('RelatingBuildingElement','RelatedOpeningElement'),'IFCRELFILLSELEMENT':('RelatingOpeningElement','RelatedBuildingElement')}
            if e.type in relation_fields:
                parent_field,child_field=relation_fields[e.type]
                for child in refs(child_field):
                    for parent in refs(parent_field):ctx['parents'][child.id].append((e.type,parent))
            if e.type in {'IFCPROJECT','IFCCONTEXT'}:
                assignment=IDS._ref(ctx,e,'UnitsInContext')
                for unit in IDS._refs(ctx,IDS._attr(ctx,assignment,'Units')):ctx['units'][IDS._attr(ctx,unit,'UnitType')]=unit
        return ctx

    @staticmethod
    def _type(ctx,entity):return ctx['types'].get(entity.id)

    @staticmethod
    def _predefined(ctx,entity):
        related=IDS._type(ctx,entity)
        selected=related or entity
        value=IDS._attr(ctx,selected,'PredefinedType')
        if value in [None,'NOTDEFINED'] and selected is not entity:
            selected=entity;value=IDS._attr(ctx,selected,'PredefinedType')
        if value=='USERDEFINED':
            custom=IDS._attr(ctx,selected,'ElementType') or IDS._attr(ctx,selected,'ProcessType') or IDS._attr(ctx,selected,'ObjectType')
            return ['USERDEFINED',custom] if custom not in [None,'','USERDEFINED'] else None
        return value

    @staticmethod
    def _classification_values(ctx,reference):
        codes=[];current=reference;seen=set()
        while current and current.id not in seen:
            seen.add(current.id)
            if current.type=='IFCCLASSIFICATION':return IDS._attr(ctx,current,'Name'),codes
            value=IDS._attr(ctx,current,'Identification') or IDS._attr(ctx,current,'ItemReference')
            if value is not None:codes.append(value)
            current=IDS._ref(ctx,current,'ReferencedSource')
        return None,codes

    @staticmethod
    def _classifications(ctx,entity):
        cache=ctx['classification_cache']
        if entity.id in cache:return cache[entity.id]
        cache[entity.id]=[]
        direct=[IDS._classification_values(ctx,r) for r in ctx['classifications'].get(entity.id,[])]
        related=IDS._type(ctx,entity)
        inherited=IDS._classifications(ctx,related) if related else []
        systems={system for system,_ in direct}
        cache[entity.id]=direct+[entry for entry in inherited if entry[0] not in systems]
        return cache[entity.id]

    @staticmethod
    def _material_values(ctx,material,seen=None):
        seen=set() if seen is None else seen
        if material.id in seen:return []
        seen.add(material.id);values=[]
        for name in ['Name','Category','LayerSetName']:
            value=IDS._attr(ctx,material,name)
            if value not in [None,'']:values.append(value)
        for arg in material.args:
            for child in IDS._refs(ctx,arg):
                if child.type.startswith('IFCMATERIAL'):values+=IDS._material_values(ctx,child,seen)
        return values

    @staticmethod
    def _properties(ctx,entity):
        cache=ctx['property_cache']
        if entity.id in cache:return cache[entity.id]
        cache[entity.id]={}
        related=IDS._type(ctx,entity)
        result={name:dict(values) for name,values in IDS._properties(ctx,related).items()} if related else {}
        for pset in ctx['psets'].get(entity.id,[]):
            name=IDS._attr(ctx,pset,'Name')
            if name is None:continue
            values={}
            props=IDS._refs(ctx,IDS._attr(ctx,pset,'HasProperties')) or IDS._refs(ctx,IDS._attr(ctx,pset,'Properties')) or IDS._refs(ctx,IDS._attr(ctx,pset,'ExtendedProperties')) or IDS._refs(ctx,IDS._attr(ctx,pset,'Quantities'))
            for prop in props:
                key=IDS._attr(ctx,prop,'Name');unit=IDS._ref(ctx,prop,'Unit')
                value=IDS._Value(None,supported=False)
                if prop.type=='IFCPROPERTYSINGLEVALUE':value=IDS._typed(IDS._attr(ctx,prop,'NominalValue'),unit)
                elif prop.type in {'IFCPROPERTYENUMERATEDVALUE','IFCPROPERTYLISTVALUE'}:
                    raw=IDS._attr(ctx,prop,'EnumerationValues' if prop.type=='IFCPROPERTYENUMERATEDVALUE' else 'ListValues') or []
                    value=IDS._Value([IDS._unwrap(v) for v in raw],next((v[1] for v in raw if isinstance(v,tuple) and v[0]=='CALL'),''),unit)
                elif prop.type=='IFCPROPERTYBOUNDEDVALUE':
                    raw=[IDS._attr(ctx,prop,k) for k in ['LowerBoundValue','UpperBoundValue','SetPointValue']];raw=[v for v in raw if v is not None]
                    value=IDS._Value([IDS._unwrap(v) for v in raw],next((v[1] for v in raw if isinstance(v,tuple) and v[0]=='CALL'),''),unit)
                elif prop.type=='IFCPROPERTYTABLEVALUE':
                    alternatives=[]
                    for column,unit_name in [('DefiningValues','DefiningUnit'),('DefinedValues','DefinedUnit')]:
                        raw=IDS._attr(ctx,prop,column) or []
                        alternatives.append(IDS._Value([IDS._unwrap(v) for v in raw],next((v[1] for v in raw if isinstance(v,tuple) and v[0]=='CALL'),''),IDS._ref(ctx,prop,unit_name)))
                    value=IDS._Value(None,alternatives=alternatives)
                elif IDS._isa(ctx,prop,'IfcPhysicalSimpleQuantity'):
                    fields=ctx['schema']['entities'][prop.type]['attributes']
                    for i,(field,kind,derived) in enumerate(fields):
                        if field.endswith('Value') and i<len(prop.args):value=IDS._Value(prop.args[i],kind.upper(),unit);break
                if key is not None:values[key]=value
            if not props and IDS._isa(ctx,pset,'IfcPreDefinedPropertySet'):
                for i,(field,kind,derived) in enumerate(ctx['schema']['entities'][pset.type]['attributes']):
                    if field not in {'GlobalId','OwnerHistory','Name','Description'} and i<len(pset.args):values[field]=IDS._Value(pset.args[i],kind.upper())
            if name in result:result[name]={**result[name],**values}
            else:result[name]=values
        cache[entity.id]=result
        return result

    @staticmethod
    def _unit_factor(ctx,unit):
        if unit is None:return 1.0
        if unit.type=='IFCSIUNIT':
            prefix=IDS._attr(ctx,unit,'Prefix');name=IDS._attr(ctx,unit,'Name')
            factor={'EXA':1e18,'PETA':1e15,'TERA':1e12,'GIGA':1e9,'MEGA':1e6,'KILO':1e3,'HECTO':1e2,'DECA':10.,'DECI':.1,'CENTI':.01,'MILLI':.001,'MICRO':1e-6,'NANO':1e-9}.get(prefix,1.)
            if name=='SQUARE_METRE':factor**=2
            if name=='CUBIC_METRE':factor**=3
            if name=='GRAM':factor*=.001
            return factor
        if IDS._isa(ctx,unit,'IfcConversionBasedUnit'):
            conversion=IDS._ref(ctx,unit,'ConversionFactor')
            return float(IDS._unwrap(IDS._attr(ctx,conversion,'ValueComponent')))*IDS._unit_factor(ctx,IDS._ref(ctx,conversion,'UnitComponent'))
        return 1.

    @staticmethod
    def _standard_value(ctx,value):
        if value.alternatives is not None:return [IDS._standard_value(ctx,v) for v in value.alternatives]
        kind=value.datatype
        mapping={'LENGTH':'LENGTHUNIT','AREA':'AREAUNIT','VOLUME':'VOLUMEUNIT','MASS':'MASSUNIT','TIME':'TIMEUNIT','PLANEANGLE':'PLANEANGLEUNIT'}
        unit_type=next((unit for name,unit in mapping.items() if kind.endswith(name+'MEASURE')),None)
        factor=IDS._unit_factor(ctx,value.unit or ctx['units'].get(unit_type)) if unit_type else 1.
        if factor==1.:return value.value
        def convert(v):return float(v)*factor if type(v) in [int,float] else v
        return [convert(v) for v in value.value] if isinstance(value.value,list) else convert(value.value)

    @staticmethod
    def _empty(value):
        return value is None or value=='' or isinstance(value,IFCLogicalUnknown) or isinstance(value,list) and not value

    @staticmethod
    def _evaluate(ctx,entity,node):
        kind=node.tag.split('}')[-1]
        cardinality=node.get('cardinality','required')
        c=lambda name:IDS._constraint(node.find(IDS.NS+name))
        matched=False;absent=False;actual=None
        if kind=='entity':
            actual=entity.type
            names=[entity.type]
            if ctx['model'].schema=='IFC2X3':
                mapping=IDS._ifc2x3_mapping()
                related=IDS._type(ctx,entity)
                names.extend(name for name,(occurrence,type_name) in mapping.items() if occurrence==entity.type and related and related.type==type_name)
            matched=any(IDS._match(c('name'),name) for name in names) and (c('predefinedType') is None or IDS._match(c('predefinedType'),IDS._predefined(ctx,entity)))
        elif kind=='attribute':
            fields=ctx['schema']['entities'].get(entity.type,{}).get('attributes',[])
            selected=[(i,field,datatype,derived) for i,(field,datatype,derived) in enumerate(fields) if IDS._match(c('name'),field)]
            raw=[entity.args[i] if i<len(entity.args) else None for i,_,_,derived in selected if not derived]
            absent=bool(raw) and all(v is None for v in raw)
            actual=raw[0] if len(raw)==1 else raw
            matched=bool(raw) and not all(IDS._empty(v) for v in raw)
            if matched and c('value') is not None:
                for i,_,datatype,derived in selected:
                    if derived:matched=False;break
                    value=entity.args[i] if i<len(entity.args) else None
                    if IDS._empty(value):continue
                    if IDS._primitive(ctx['schema'],datatype) in {'aggregate','select'} or isinstance(value,tuple):matched=False;break
                    if not IDS._match(c('value'),value):matched=False;break
        elif kind=='classification':
            entries=IDS._classifications(ctx,entity);absent=not entries;actual=entries
            matched=any(IDS._match(c('system'),system) and (c('value') is None or any(IDS._match(c('value'),code) for code in codes)) for system,codes in entries)
        elif kind=='material':
            materials=ctx['materials'].get(entity.id,[])
            related=IDS._type(ctx,entity)
            if not materials and related:materials=ctx['materials'].get(related.id,[])
            absent=not materials;actual=[v for material in materials for v in IDS._material_values(ctx,material)]
            matched=bool(materials) and (c('value') is None or any(IDS._match(c('value'),v) for v in actual))
        elif kind=='property':
            psets=IDS._properties(ctx,entity)
            selected=[props for name,props in psets.items() if IDS._match(c('propertySet'),name)]
            absent=not selected
            values=[];matched=bool(selected)
            for props in selected:
                matching=[v for name,v in props.items() if IDS._match(c('baseName'),name)]
                if not matching:matched=False
                values+=matching
            for value in values:
                if value.alternatives is not None:
                    candidates=[v for v in value.alternatives if not node.get('dataType') or v.datatype.upper()==node.get('dataType').upper()]
                    if not any(not IDS._empty(IDS._standard_value(ctx,v)) and IDS._match(c('value'),IDS._standard_value(ctx,v)) for v in candidates):matched=False
                    continue
                actual_value=IDS._standard_value(ctx,value)
                if not value.supported or IDS._empty(actual_value):matched=False;continue
                requested=node.get('dataType')
                if requested and requested.upper()!=value.datatype.upper():matched=False;continue
                if not IDS._match(c('value'),actual_value):matched=False
            actual=[IDS._standard_value(ctx,v) for v in values]
            if len(actual)==1:actual=actual[0]
            # An absent property is optional even inside an existing pset.
            if not values:absent=True
        elif kind=='partOf':
            relation=node.get('relation');target=relation.upper() if relation else None
            queue=deque([(entity,False)]);visited=set();candidates=[]
            while queue:
                current,seen_relation=queue.popleft()
                if (current.id,seen_relation) in visited:continue
                visited.add((current.id,seen_relation))
                for rel,parent in ctx['parents'].get(current.id,[]):
                    if target is None and rel=='IFCRELASSIGNSTOGROUP':continue
                    if target in {'IFCRELAGGREGATES','IFCRELNESTS','IFCRELASSIGNSTOGROUP'} and rel!=target:continue
                    seen=seen_relation or target is None or rel==target
                    if seen:candidates.append(parent)
                    queue.append((parent,seen))
            required=node.find(IDS.NS+'entity');actual=[parent.type for parent in candidates];absent=not candidates
            matched=any(IDS._evaluate(ctx,parent,required)[0] for parent in candidates)
        else:raise ValueError('The IDS facet is not supported: '+kind)
        if cardinality=='prohibited':matched=not matched
        elif cardinality=='optional' and absent:matched=True
        return matched,actual

    @staticmethod
    def _describe(node):
        kind=node.tag.split('}')[-1]
        parts=[]
        for child in node:
            if child.tag==IDS.NS+'entity':parts.append(IDS._describe(child));continue
            constraint=IDS._constraint(child)
            if constraint is not None:parts.append(child.tag.split('}')[-1]+'='+str(constraint.get('simple',dict(constraint))))
        return kind+': '+', '.join(parts)

    @staticmethod
    def Validate(model,document):
        """Return specification outcomes and traceable element-level findings."""
        if model is None or not model.schema:raise ValueError('The IFC could not be read. Supply an uncompressed IFC STEP file.')
        ctx=IDS._context(model)
        specifications=[];findings=[];unique=set();failed=set();checks=passed_checks=0;warnings=[]
        for number,spec in enumerate(document.findall(IDS.NS+'specifications/'+IDS.NS+'specification'),1):
            sid=f'S{number:03d}';identifier=spec.get('identifier') or sid;name=spec.get('name') or 'Unnamed specification'
            app=spec.find(IDS.NS+'applicability');requirements=list(spec.find(IDS.NS+'requirements') or [])
            minimum=int(app.get('minOccurs','1'));maximum=app.get('maxOccurs','unbounded')
            prohibited=maximum=='0';cardinality='prohibited' if prohibited else 'required' if minimum else 'optional'
            if prohibited and requirements:raise ValueError('The IDS is invalid or unsupported: a prohibited specification must not contain requirements.')
            versions=spec.get('ifcVersion','').split();version_match=model.schema in versions
            if not version_match:warnings.append(f'{identifier}: IDS declares {", ".join(versions)}; model uses {model.schema}. Requirements were assessed using the actual model schema.')
            applicable=[entity for entity in model.entities.values() if all(IDS._evaluate(ctx,entity,node)[0] for node in app)]
            unique.update(e.id for e in applicable);bad=set()
            for entity in applicable:
                for requirement in requirements:
                    checks+=1;matched,actual=IDS._evaluate(ctx,entity,requirement)
                    if matched:passed_checks+=1;continue
                    bad.add(entity.id);label=IDS._describe(requirement)
                    raw_name=IDS._attr(ctx,entity,'Name') or '';tag=IDS._attr(ctx,entity,'Tag') or ''
                    global_id=IDS._attr(ctx,entity,'GlobalId') or ''
                    value_constraint=IDS._constraint(requirement.find(IDS.NS+'value'))
                    expected=value_constraint.get('simple') if value_constraint and 'simple' in value_constraint else label
                    findings.append({'id':f'F{len(findings)+1:05d}','specification':sid,'identifier':identifier,'specification_name':name,
                        'requirement':label,'label':label,'facet':requirement.tag.split('}')[-1].title(),'expected':expected,
                        'actual':IDS._display(actual),'reason':'Required information is missing.' if IDS._empty(actual) else 'The supplied information does not satisfy the IDS requirement.',
                        'correction':requirement.get('instructions') or spec.get('instructions') or label,
                        'global_id':global_id,'step_id':entity.id,'name':raw_name,'tag':tag,'ifc_class':ctx['schema']['entities'].get(entity.type,{}).get('name',entity.type)})
            if prohibited:bad={e.id for e in applicable}
            fail=bool(bad) or minimum>0 and not applicable
            if fail and not any(f['specification']==sid for f in findings):
                for entity in applicable if prohibited else [None]:
                    findings.append({'id':f'F{len(findings)+1:05d}','specification':sid,'identifier':identifier,'specification_name':name,
                        'requirement':'Specification applicability','label':'Applicability','facet':'Applicability','expected':cardinality,
                        'actual':str(len(applicable))+' applicable elements','reason':'Prohibited elements are present.' if prohibited else 'No elements match this required specification.',
                        'correction':spec.get('instructions') or 'Review the IDS applicability and the delivery with the model author.',
                        'global_id':IDS._attr(ctx,entity,'GlobalId') or '','step_id':entity.id if entity else '',
                        'name':IDS._attr(ctx,entity,'Name') or '','tag':IDS._attr(ctx,entity,'Tag') or '','ifc_class':entity.type if entity else ''})
            failed.update(bad)
            specifications.append({'id':sid,'identifier':identifier,'name':name,'description':spec.get('description',''),'instructions':spec.get('instructions',''),
                'status':'fail' if fail else 'pass' if applicable else 'no_applicable','applicable':len(applicable),'passed':len(applicable)-len(bad),'failed':len(bad),
                'applicability':[IDS._describe(node) for node in app],'cardinality':cardinality,'ifc_versions':versions,'version_matches':version_match,
                'requirements':[IDS._describe(node) for node in requirements]})
        return {'status':'fail' if any(s['status']=='fail' for s in specifications) else 'pass','specifications':specifications,'findings':findings,'warnings':warnings,
            'summary':{'specifications':len(specifications),'passed_specifications':sum(s['status']=='pass' for s in specifications),
            'failed_specifications':sum(s['status']=='fail' for s in specifications),'incomplete_specifications':0,'no_applicable_specifications':sum(s['status']=='no_applicable' for s in specifications),
            'elements':len(unique),'failed_elements':len(failed),'passed_elements':len(unique-failed),'checks':checks,'passed_checks':passed_checks,'failed_checks':checks-passed_checks,'findings':len(findings)}}

    @staticmethod
    def _display(value):
        if IDS._empty(value):return 'Missing'
        if type(value) is bool:return 'TRUE' if value else 'FALSE'
        return str(value)

    @staticmethod
    def ValidateFiles(ifcPath,idsPath):
        """Convenience API for non-web applications."""
        return IDS.Validate(IFC.ModelByPath(str(ifcPath),silent=True),IDS.ByPath(idsPath))
